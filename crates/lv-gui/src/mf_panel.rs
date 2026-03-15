//! MatrixForge pipeline panel.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use mf_pipeline::pipeline::run_mf_pipeline;
use mf_pipeline::types::{MfConfig, MfPipelineConfig, SimToDistMethod};

use crate::state::{AppState, PipelineEvent, PipelineJob};

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
    output_dir: Option<PathBuf>,
    /// Set to true when user clicks "Send output to AlignSpace".
    pub send_to_as: bool,
    status: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SimToDistMethodUi {
    Linear,
    Cosine,
    Info,
}

impl Default for MfPanel {
    fn default() -> Self {
        Self {
            input_files: vec![],
            input_dir: None,
            window_size: 5,
            slide_rate: 1,
            language: "en".into(),
            use_pmi: true,
            min_count: 2,
            min_pmi: 0.0,
            sim_to_dist: SimToDistMethodUi::Linear,
            output_dir: None,
            send_to_as: false,
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
            if ui.button("Add directory…").clicked() {
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

        ui.separator();

        // ── Config ─────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Window:");
            ui.add(egui::Slider::new(&mut self.window_size, 2..=20));
        });
        ui.horizontal(|ui| {
            ui.label("Slide rate:");
            ui.add(egui::DragValue::new(&mut self.slide_rate).range(1..=self.window_size));
        });
        ui.horizontal(|ui| {
            ui.label("Language:");
            ui.text_edit_singleline(&mut self.language);
        });
        ui.horizontal(|ui| {
            ui.label("Min count:");
            ui.add(egui::DragValue::new(&mut self.min_count).range(1..=1000u64));
        });
        ui.horizontal(|ui| {
            ui.label("Min PMI:");
            ui.add(
                egui::DragValue::new(&mut self.min_pmi)
                    .speed(0.01)
                    .range(0.0..=10.0)
                    .fixed_decimals(3),
            );
        });
        ui.checkbox(&mut self.use_pmi, "Use PMI (vs raw counts)");

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
                    Ok(path) => {
                        self.status = format!("Done: {}", path.display());
                        if ui.button("Send output to AlignSpace").clicked() {
                            self.send_to_as = true;
                        }
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
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let mut sources: Vec<PathBuf> = self.input_files.clone();
        if let Some(ref dir) = self.input_dir {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.is_file() {
                        sources.push(p);
                    }
                }
            }
        }
        let output_dir = self.output_dir.clone();

        let mf_config = MfConfig {
            window_size: self.window_size,
            slide_rate: self.slide_rate,
            language: self.language.clone(),
            use_pmi: self.use_pmi,
            min_count: self.min_count,
            min_pmi: self.min_pmi,
            unicode_normalize: true,
            sim_to_dist: match self.sim_to_dist {
                SimToDistMethodUi::Linear => SimToDistMethod::Linear,
                SimToDistMethodUi::Cosine => SimToDistMethod::Cosine,
                SimToDistMethodUi::Info => SimToDistMethod::Info,
            },
        };
        let pipeline_cfg = MfPipelineConfig {
            input_paths: sources,
            output_dir: output_dir.clone(),
            mf_config,
            write_json: true,
            write_xlsx: false,
        };

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Running MatrixForge…".into(),
                pct: 0.2,
            });
            match run_mf_pipeline(&pipeline_cfg) {
                Ok(_output) => {
                    let _ = tx.send(PipelineEvent::Progress {
                        step: "Done.".into(),
                        pct: 1.0,
                    });
                    // Report the output directory as the result path
                    let out_path = output_dir
                        .unwrap_or_else(|| PathBuf::from("."))
                        .join("mf_output.json");
                    let _ = tx.send(PipelineEvent::Done(Ok(out_path)));
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
}
