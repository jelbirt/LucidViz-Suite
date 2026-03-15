//! AlignSpace pipeline panel.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use as_pipeline::pipeline::run_pipeline;
use as_pipeline::types::{
    AsPipelineInput, MdsConfig, MdsDimMode, ProcrustesMode, SmacofConfig, SmacofInit,
};
use lv_data::EtvDataset;
use ndarray::Array2;

use crate::state::{AppState, PipelineEvent, PipelineJob};

/// Panel for configuring and launching the AlignSpace MDS pipeline.
pub struct AsPanel {
    // MDS config
    algorithm: MdsAlgorithmUi,
    dim_mode: MdsDimModeUi,
    procrustes: ProcrustesMode,
    normalize: bool,
    norm_range: f64,
    // SMACOF sub-config
    smacof_max_iter: u32,
    smacof_tol: f64,
    smacof_random: bool,
    smacof_seed: u64,
    // Output dir
    output_dir: Option<PathBuf>,
    status: String,
    /// Set to true when user clicks "Load output into LV".
    pub load_output: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MdsAlgorithmUi {
    Auto,
    Classical,
    Smacof,
    Pivot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MdsDimModeUi {
    Maximum,
    Visual,
    Two,
    Three,
}

impl Default for AsPanel {
    fn default() -> Self {
        Self {
            algorithm: MdsAlgorithmUi::Auto,
            dim_mode: MdsDimModeUi::Three,
            procrustes: ProcrustesMode::None,
            normalize: true,
            norm_range: 300.0,
            smacof_max_iter: 300,
            smacof_tol: 1e-6,
            smacof_random: false,
            smacof_seed: 42,
            output_dir: None,
            status: "Ready.".into(),
            load_output: false,
        }
    }
}

impl AsPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.heading("AlignSpace");
        ui.separator();

        // ── Algorithm ──────────────────────────────────────────────────────
        egui::ComboBox::from_label("Algorithm")
            .selected_text(format!("{:?}", self.algorithm))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.algorithm, MdsAlgorithmUi::Auto, "Auto");
                ui.selectable_value(&mut self.algorithm, MdsAlgorithmUi::Classical, "Classical");
                ui.selectable_value(&mut self.algorithm, MdsAlgorithmUi::Smacof, "SMACOF");
                ui.selectable_value(&mut self.algorithm, MdsAlgorithmUi::Pivot, "Pivot");
            });

        // SMACOF sub-config
        if self.algorithm == MdsAlgorithmUi::Smacof {
            ui.indent("smacof", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Max iter:");
                    ui.add(egui::DragValue::new(&mut self.smacof_max_iter).range(10..=5000));
                });
                ui.horizontal(|ui| {
                    ui.label("Tolerance:");
                    ui.add(
                        egui::DragValue::new(&mut self.smacof_tol)
                            .speed(1e-7)
                            .range(1e-10..=1e-1)
                            .fixed_decimals(8),
                    );
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.smacof_random, "Random init");
                    if self.smacof_random {
                        ui.label("Seed:");
                        ui.add(egui::DragValue::new(&mut self.smacof_seed));
                    }
                });
            });
        }

        // ── Dimensionality ─────────────────────────────────────────────────
        egui::ComboBox::from_label("Dimensions")
            .selected_text(self.dim_mode_label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.dim_mode, MdsDimModeUi::Three, "3D");
                ui.selectable_value(&mut self.dim_mode, MdsDimModeUi::Two, "2D (Visual)");
                ui.selectable_value(&mut self.dim_mode, MdsDimModeUi::Maximum, "Maximum");
            });

        // ── Procrustes ─────────────────────────────────────────────────────
        egui::ComboBox::from_label("Procrustes")
            .selected_text(format!("{:?}", self.procrustes))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.procrustes, ProcrustesMode::None, "None");
                ui.selectable_value(
                    &mut self.procrustes,
                    ProcrustesMode::TimeSeries,
                    "Time Series",
                );
                ui.selectable_value(
                    &mut self.procrustes,
                    ProcrustesMode::OptimalChoice,
                    "Optimal Choice",
                );
            });

        // ── Normalise ──────────────────────────────────────────────────────
        ui.checkbox(&mut self.normalize, "Normalize coordinates");
        if self.normalize {
            ui.horizontal(|ui| {
                ui.label("Range:");
                ui.add(
                    egui::DragValue::new(&mut self.norm_range)
                        .speed(1.0)
                        .range(1.0..=10000.0),
                );
            });
        }

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
        let can_run = state.dataset.is_some()
            && state
                .as_job
                .as_ref()
                .map(|j| j.result.is_some())
                .unwrap_or(true);

        if ui
            .add_enabled(can_run, egui::Button::new("Run AlignSpace"))
            .clicked()
        {
            let ds_clone = state.dataset.clone();
            if let Some(ds) = ds_clone {
                let out = self
                    .output_dir
                    .clone()
                    .unwrap_or_else(|| PathBuf::from("."));
                self.launch_job(&ds, out, state);
            }
        }

        // ── Progress ───────────────────────────────────────────────────────
        if let Some(job) = &state.as_job {
            if job.result.is_none() {
                ui.add(egui::ProgressBar::new(job.last_pct).text(&job.last_step));
            } else if let Some(result) = &job.result {
                match result {
                    Ok(path) => {
                        self.status = format!("Done: {}", path.display());
                        if ui.button("Load output into LV").clicked() {
                            self.load_output = true;
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

    fn dim_mode_label(&self) -> &'static str {
        match self.dim_mode {
            MdsDimModeUi::Three => "3D",
            MdsDimModeUi::Two => "2D (Visual)",
            MdsDimModeUi::Maximum => "Maximum",
            MdsDimModeUi::Visual => "2D (Visual)",
        }
    }

    fn build_mds_config(&self) -> MdsConfig {
        let init = if self.smacof_random {
            SmacofInit::Random(self.smacof_seed)
        } else {
            SmacofInit::Classical
        };
        match self.algorithm {
            MdsAlgorithmUi::Auto => MdsConfig::Auto,
            MdsAlgorithmUi::Classical => MdsConfig::Classical,
            MdsAlgorithmUi::Smacof => MdsConfig::Smacof(SmacofConfig {
                max_iter: self.smacof_max_iter,
                tolerance: self.smacof_tol,
                init,
            }),
            MdsAlgorithmUi::Pivot => MdsConfig::PivotMds { n_pivots: 50 },
        }
    }

    fn launch_job(&self, ds: &EtvDataset, output_dir: PathBuf, state: &mut AppState) {
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        // Collect labels and build adjacency matrices from each sheet
        let all_labels: Vec<String> = if let Some(first) = ds.sheets.first() {
            first.rows.iter().map(|r| r.label.clone()).collect()
        } else {
            vec![]
        };
        let n = all_labels.len();

        let datasets: Vec<(String, Array2<f64>)> = ds
            .sheets
            .iter()
            .map(|sheet| {
                let mut mat = Array2::<f64>::zeros((n, n));
                for edge in &sheet.edges {
                    let i = all_labels.iter().position(|l| l == &edge.from);
                    let j = all_labels.iter().position(|l| l == &edge.to);
                    if let (Some(i), Some(j)) = (i, j) {
                        mat[[i, j]] = edge.strength;
                        mat[[j, i]] = edge.strength;
                    }
                }
                (sheet.name.clone(), mat)
            })
            .collect();

        let mds_config = self.build_mds_config();
        let dim_mode = match self.dim_mode {
            MdsDimModeUi::Three => MdsDimMode::Fixed(3),
            MdsDimModeUi::Two => MdsDimMode::Visual,
            MdsDimModeUi::Maximum => MdsDimMode::Maximum,
            MdsDimModeUi::Visual => MdsDimMode::Visual,
        };
        let procrustes_mode = self.procrustes;
        let normalize = self.normalize;
        let target_range = self.norm_range;

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Computing MDS…".into(),
                pct: 0.1,
            });
            let input = AsPipelineInput {
                datasets,
                labels: all_labels,
                mds_config,
                procrustes_mode,
                mds_dims: dim_mode,
                normalize,
                target_range,
                procrustes_scale: true,
            };
            match run_pipeline(&input) {
                Ok(result) => {
                    let _ = tx.send(PipelineEvent::Progress {
                        step: "Writing output…".into(),
                        pct: 0.9,
                    });
                    // Serialize result to JSON in output_dir
                    let out_file = output_dir.join("as_result.json");
                    if let Ok(json) = serde_json::to_string(&result.etv_dataset) {
                        let _ = std::fs::write(&out_file, json);
                    }
                    let _ = tx.send(PipelineEvent::Done(Ok(out_file)));
                }
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Done(Err(e.to_string())));
                }
            }
        });

        state.as_job = Some(PipelineJob {
            receiver: rx,
            last_step: "Starting…".into(),
            last_pct: 0.0,
            result: None,
        });
    }
}
