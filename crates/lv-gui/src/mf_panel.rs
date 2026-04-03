//! MatrixForge pipeline panel.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use mf_pipeline::output::{read_mf_json, read_mf_series_json};
use mf_pipeline::pipeline::{run_mf_pipeline, run_mf_series_pipeline};
use mf_pipeline::types::{
    MfConfig, MfOutput, MfPipelineConfig, MfSeriesOutput, MfSliceMode, SimToDistMethod,
    SimilarityMethod,
};

use crate::state::{AppState, AsInputSource, PipelineEvent, PipelineJob, PipelineOutcome};

/// Panel for configuring and launching the MatrixForge text-analysis pipeline.
pub struct MfPanel {
    input_files: Vec<PathBuf>,
    input_dir: Option<PathBuf>,
    window_size: usize,
    slide_rate: usize,
    language: String,
    use_pmi: bool,
    min_count: u64,
    min_pmi: f64,
    sim_to_dist: SimToDistMethodUi,
    slice_mode: MfSliceModeUi,
    slice_size: usize,
    unicode_normalize: bool,
    shared_vocabulary: bool,
    min_tokens_per_slice: usize,
    similarity_method: SimilarityMethodUi,
    output_dir: Option<PathBuf>,
    status: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimilarityMethodUi {
    Nppmi,
    PpmiSvd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimToDistMethodUi {
    Linear,
    Cosine,
    Info,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MfSliceModeUi {
    None,
    PerFile,
    FixedTokenBatch,
}

impl Default for MfPanel {
    fn default() -> Self {
        let defaults = MfConfig::default();
        Self {
            input_files: vec![],
            input_dir: None,
            window_size: defaults.window_size,
            slide_rate: defaults.slide_rate,
            language: defaults.language,
            use_pmi: defaults.use_pmi,
            min_count: defaults.min_count,
            min_pmi: defaults.min_pmi,
            sim_to_dist: SimToDistMethodUi::from(defaults.sim_to_dist),
            slice_mode: MfSliceModeUi::from(defaults.slice_mode),
            slice_size: defaults.slice_size,
            unicode_normalize: defaults.unicode_normalize,
            shared_vocabulary: defaults.shared_vocabulary,
            min_tokens_per_slice: defaults.min_tokens_per_slice,
            similarity_method: match defaults.similarity_method {
                SimilarityMethod::Nppmi => SimilarityMethodUi::Nppmi,
                SimilarityMethod::PpmiSvd => SimilarityMethodUi::PpmiSvd,
            },
            output_dir: None,
            status: "Ready.".into(),
        }
    }
}

impl MfPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.heading("MatrixForge");
        ui.separator();

        // ── Input ──────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("Add files…").clicked() {
                if let Some(files) = rfd::FileDialog::new()
                    .add_filter("Text", &["txt", "csv", "md"])
                    .pick_files()
                {
                    self.input_files.extend(files);
                }
            }
            if ui.button("Add .txt directory…").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    self.input_dir = Some(dir);
                }
            }
            if ui.button("Clear").clicked() {
                self.input_files.clear();
                self.input_dir = None;
            }
        });

        let total = self.input_files.len() + self.input_dir.as_ref().map(|_| 1).unwrap_or(0);
        ui.label(format!("{total} source(s) selected"));
        ui.small(
            "Directory ingestion is non-recursive and currently loads only top-level .txt files. Add .csv/.md inputs individually.",
        );

        ui.separator();

        // ── Config ─────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Window:").on_hover_text(
                "Co-occurrence context window size (tokens on each side). \
                 Larger windows capture broader semantic relationships. Typical: 5-10.",
            );
            ui.add(egui::Slider::new(&mut self.window_size, 2..=20));
        });
        ui.horizontal(|ui| {
            ui.label("Slide rate:").on_hover_text(
                "Step size for sliding the context window. \
                 1 = every position, higher = skip positions (faster, less precise).",
            );
            ui.add(egui::DragValue::new(&mut self.slide_rate).range(1..=self.window_size));
        });
        ui.horizontal(|ui| {
            ui.label("Language:").on_hover_text(
                "BCP 47 language code for stop-word filtering. \
                 Examples: 'en' (English), 'de' (German), 'fr' (French).",
            );
            ui.text_edit_singleline(&mut self.language);
        });
        ui.horizontal(|ui| {
            ui.label("Min count:").on_hover_text(
                "Minimum co-occurrence count for a word pair to be included. \
                 Higher values filter rare co-occurrences (reduces noise).",
            );
            ui.add(egui::DragValue::new(&mut self.min_count).range(1..=1000u64));
        });
        ui.horizontal(|ui| {
            ui.label("Min PMI:").on_hover_text(
                "Minimum PMI threshold for an edge in the co-occurrence graph. \
                 Higher values retain only strongly associated word pairs. Typical: 0.0-2.0.",
            );
            ui.add(
                egui::DragValue::new(&mut self.min_pmi)
                    .speed(0.01)
                    .range(0.0..=10.0)
                    .fixed_decimals(3),
            );
        });
        ui.checkbox(&mut self.use_pmi, "Use PMI (vs raw counts)")
            .on_hover_text(
                "When enabled, uses Pointwise Mutual Information to weight co-occurrences. \
                 PMI highlights statistically significant associations over raw frequency.",
            );

        if self.use_pmi {
            egui::ComboBox::from_label("Similarity method")
                .selected_text(match self.similarity_method {
                    SimilarityMethodUi::Nppmi => "NPPMI (raw)",
                    SimilarityMethodUi::PpmiSvd => "PPMI + SVD (denoised)",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.similarity_method,
                        SimilarityMethodUi::Nppmi,
                        "NPPMI (raw)",
                    );
                    ui.selectable_value(
                        &mut self.similarity_method,
                        SimilarityMethodUi::PpmiSvd,
                        "PPMI + SVD (denoised)",
                    );
                });
        }

        egui::CollapsingHeader::new("Advanced options")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.unicode_normalize, "Unicode normalize (NFC)");
                ui.checkbox(
                    &mut self.shared_vocabulary,
                    "Shared vocabulary across slices",
                );
                ui.horizontal(|ui| {
                    ui.label("Min tokens per slice:");
                    ui.add(
                        egui::DragValue::new(&mut self.min_tokens_per_slice).range(1..=1_000_000),
                    );
                });
            });

        egui::ComboBox::from_label("Slice mode")
            .selected_text(match self.slice_mode {
                MfSliceModeUi::None => "Single matrix",
                MfSliceModeUi::PerFile => "Per file",
                MfSliceModeUi::FixedTokenBatch => "Fixed token batch",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.slice_mode, MfSliceModeUi::None, "Single matrix");
                ui.selectable_value(&mut self.slice_mode, MfSliceModeUi::PerFile, "Per file");
                ui.selectable_value(
                    &mut self.slice_mode,
                    MfSliceModeUi::FixedTokenBatch,
                    "Fixed token batch",
                );
            });

        if self.slice_mode == MfSliceModeUi::FixedTokenBatch {
            ui.horizontal(|ui| {
                ui.label("Batch size:");
                ui.add(egui::DragValue::new(&mut self.slice_size).range(10..=100_000));
            });
        }

        egui::ComboBox::from_label("Sim→Dist")
            .selected_text(format!("{:?}", self.sim_to_dist))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.sim_to_dist, SimToDistMethodUi::Linear, "Linear");
                ui.selectable_value(&mut self.sim_to_dist, SimToDistMethodUi::Cosine, "Cosine");
                ui.selectable_value(&mut self.sim_to_dist, SimToDistMethodUi::Info, "Info");
            });

        // ── Output dir ─────────────────────────────────────────────────────
        ui.separator();
        ui.horizontal(|ui| {
            let dir_label = self
                .output_dir
                .as_ref()
                .and_then(|p| p.to_str())
                .unwrap_or("<none>");
            ui.label(format!("Output: {dir_label}"));
            if ui.button("Browse…").clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.output_dir = Some(p);
                }
            }
        });

        // ── Run button ─────────────────────────────────────────────────────
        let has_input = !self.input_files.is_empty() || self.input_dir.is_some();
        let can_run = has_input
            && state
                .mf_job
                .as_ref()
                .map(|j| j.result.is_some())
                .unwrap_or(true);

        if ui
            .add_enabled(can_run, egui::Button::new("Run MatrixForge"))
            .clicked()
        {
            self.launch_job(state);
        }

        // ── Progress ───────────────────────────────────────────────────────
        if let Some(job) = &state.mf_job {
            if job.result.is_none() {
                ui.add(egui::ProgressBar::new(job.last_pct).text(&job.last_step));
            } else if let Some(result) = &job.result {
                match result {
                    Ok(PipelineOutcome::File(path)) => {
                        self.status = format!("Done: {}", path.display());
                        if ui.button("Send output to AlignSpace").clicked() {
                            match self.slice_mode {
                                MfSliceModeUi::None => match load_mf_output(path) {
                                    Ok(output) => {
                                        state.mf_output = Some(output);
                                        state.mf_series_output = None;
                                        state.as_input_source = AsInputSource::MatrixForge;
                                        state.load_error = None;
                                        self.status =
                                            "MatrixForge output is ready in AlignSpace.".into();
                                    }
                                    Err(e) => {
                                        state.load_error = Some(e.clone());
                                        self.status = format!("Error: {e}");
                                    }
                                },
                                _ => match load_mf_series_output(path) {
                                    Ok(output) => {
                                        state.mf_series_output = Some(output);
                                        state.mf_output = None;
                                        state.as_input_source = AsInputSource::MatrixForgeSeries;
                                        state.load_error = None;
                                        self.status =
                                            "MatrixForge series is ready in AlignSpace.".into();
                                    }
                                    Err(e) => {
                                        state.load_error = Some(e.clone());
                                        self.status = format!("Error: {e}");
                                    }
                                },
                            }
                        }
                    }
                    Ok(PipelineOutcome::AsRun(_)) => {
                        self.status = "Unexpected pipeline output type.".into();
                    }
                    Err(e) => {
                        self.status = format!("Error: {e}");
                    }
                }
            }
        }
        ui.label(&self.status);
    }

    fn launch_job(&mut self, state: &mut AppState) {
        // Drop any previous job. If the old thread is still running, its
        // channel sends will fail silently on the disconnected receiver,
        // causing it to stop naturally after completing its current step.
        state.mf_job = None;

        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let mut sources: Vec<PathBuf> = self.input_files.clone();
        if let Some(ref dir) = self.input_dir {
            sources.push(dir.clone());
        }
        let output_dir = self
            .output_dir
            .clone()
            .unwrap_or_else(default_mf_output_dir);

        let mf_config = self.build_mf_config();
        let pipeline_cfg = MfPipelineConfig {
            input_paths: sources,
            output_dir: Some(output_dir.clone()),
            mf_config,
            write_json: true,
            write_xlsx: false,
        };

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Running MatrixForge…".into(),
                pct: 0.2,
            });
            let is_series = pipeline_cfg.mf_config.slice_mode != MfSliceMode::None;
            let result = if is_series {
                run_mf_series_pipeline(&pipeline_cfg).map(|_| ())
            } else {
                run_mf_pipeline(&pipeline_cfg).map(|_| ())
            };
            match result {
                Ok(_output) => {
                    let _ = tx.send(PipelineEvent::Progress {
                        step: "Done.".into(),
                        pct: 1.0,
                    });
                    let out_path = if is_series {
                        output_dir.join("mf_series_output.json")
                    } else {
                        output_dir.join("mf_output.json")
                    };
                    let _ = tx.send(PipelineEvent::Done(Ok(PipelineOutcome::File(out_path))));
                }
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Done(Err(e.to_string())));
                }
            }
        });

        state.mf_job = Some(PipelineJob {
            receiver: rx,
            last_step: "Starting…".into(),
            last_pct: 0.0,
            result: None,
        });
    }

    fn build_mf_config(&self) -> MfConfig {
        MfConfig {
            window_size: self.window_size,
            slide_rate: self.slide_rate,
            language: self.language.clone(),
            use_pmi: self.use_pmi,
            min_count: self.min_count,
            min_pmi: self.min_pmi,
            unicode_normalize: self.unicode_normalize,
            sim_to_dist: match self.sim_to_dist {
                SimToDistMethodUi::Linear => SimToDistMethod::Linear,
                SimToDistMethodUi::Cosine => SimToDistMethod::Cosine,
                SimToDistMethodUi::Info => SimToDistMethod::Info,
            },
            slice_mode: match self.slice_mode {
                MfSliceModeUi::None => MfSliceMode::None,
                MfSliceModeUi::PerFile => MfSliceMode::PerFile,
                MfSliceModeUi::FixedTokenBatch => MfSliceMode::FixedTokenBatch,
            },
            slice_size: self.slice_size,
            min_tokens_per_slice: self.min_tokens_per_slice,
            shared_vocabulary: self.shared_vocabulary,
            similarity_method: match self.similarity_method {
                SimilarityMethodUi::Nppmi => SimilarityMethod::Nppmi,
                SimilarityMethodUi::PpmiSvd => SimilarityMethod::PpmiSvd,
            },
        }
    }
}

impl From<SimToDistMethod> for SimToDistMethodUi {
    fn from(value: SimToDistMethod) -> Self {
        match value {
            SimToDistMethod::Linear => Self::Linear,
            SimToDistMethod::Cosine => Self::Cosine,
            SimToDistMethod::Info => Self::Info,
        }
    }
}

impl From<MfSliceMode> for MfSliceModeUi {
    fn from(value: MfSliceMode) -> Self {
        match value {
            MfSliceMode::None => Self::None,
            MfSliceMode::PerFile => Self::PerFile,
            MfSliceMode::FixedTokenBatch => Self::FixedTokenBatch,
        }
    }
}

fn default_mf_output_dir() -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir()
        .join("lucid-viz")
        .join("matrixforge")
        .join(stamp.to_string())
}

fn load_mf_output(path: &std::path::Path) -> Result<MfOutput, String> {
    read_mf_json(path).map_err(|e| e.to_string())
}

fn load_mf_series_output(path: &std::path::Path) -> Result<MfSeriesOutput, String> {
    read_mf_series_json(path).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::MfPanel;
    use mf_pipeline::types::MfConfig;

    #[test]
    fn default_panel_matches_mf_defaults() {
        let panel = MfPanel::default();
        let defaults = MfConfig::default();
        let built = panel.build_mf_config();

        assert_eq!(built.window_size, defaults.window_size);
        assert_eq!(built.slide_rate, defaults.slide_rate);
        assert_eq!(built.language, defaults.language);
        assert_eq!(built.use_pmi, defaults.use_pmi);
        assert_eq!(built.min_count, defaults.min_count);
        assert_eq!(built.min_pmi, defaults.min_pmi);
        assert_eq!(built.unicode_normalize, defaults.unicode_normalize);
        assert_eq!(built.sim_to_dist, defaults.sim_to_dist);
        assert_eq!(built.slice_mode, defaults.slice_mode);
        assert_eq!(built.slice_size, defaults.slice_size);
        assert_eq!(built.min_tokens_per_slice, defaults.min_tokens_per_slice);
        assert_eq!(built.shared_vocabulary, defaults.shared_vocabulary);
    }
}
