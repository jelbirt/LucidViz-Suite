//! AlignSpace pipeline panel.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use as_pipeline::pipeline::{mf_output_to_distance_matrix, run_distance_pipeline, run_pipeline};
use as_pipeline::types::{
    AsDistancePipelineInput, AsPipelineInput, MdsConfig, MdsDimMode, ProcrustesMode, SmacofConfig,
    SmacofInit,
};
use lv_data::schema::EtvDataset;
use mf_pipeline::pipeline::mf_series_output_to_as_input;
use ndarray::Array2;

use crate::state::{AppState, AsInputSource, PipelineEvent, PipelineJob};

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
    Planar,
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
                ui.selectable_value(&mut self.dim_mode, MdsDimModeUi::Planar, "2D");
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
        let has_input = match state.as_input_source {
            AsInputSource::Dataset => state.dataset.is_some(),
            AsInputSource::MatrixForge => state.mf_output.is_some(),
            AsInputSource::MatrixForgeSeries => state.mf_series_output.is_some(),
        };

        let can_run = has_input
            && state
                .as_job
                .as_ref()
                .map(|j| j.result.is_some())
                .unwrap_or(true);

        if ui
            .add_enabled(can_run, egui::Button::new("Run AlignSpace"))
            .clicked()
        {
            let out = self
                .output_dir
                .clone()
                .unwrap_or_else(|| PathBuf::from("."));

            match state.as_input_source {
                AsInputSource::Dataset => {
                    if let Some(ds) = state.dataset.clone() {
                        self.launch_dataset_job(&ds, out, state);
                    }
                }
                AsInputSource::MatrixForge => {
                    if let Some(mf_output) = state.mf_output.clone() {
                        self.launch_mf_job(&mf_output, out, state);
                    }
                }
                AsInputSource::MatrixForgeSeries => {
                    if let Some(series) = state.mf_series_output.clone() {
                        self.launch_mf_series_job(&series, out, state);
                    }
                }
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
                            match load_etv_dataset(path) {
                                Ok(dataset) => {
                                    state.dataset = Some(dataset);
                                    state.source_path = Some(path.clone());
                                    state.as_input_source = AsInputSource::Dataset;
                                    state.dataset_changed = true;
                                    state.load_error = None;
                                    self.status = "AlignSpace output loaded into LV.".into();
                                }
                                Err(e) => {
                                    state.load_error = Some(e.clone());
                                    self.status = format!("Error: {e}");
                                }
                            }
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
            MdsDimModeUi::Planar => "2D",
            MdsDimModeUi::Maximum => "Maximum",
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

    fn selected_dim_mode(&self) -> MdsDimMode {
        match self.dim_mode {
            MdsDimModeUi::Three => MdsDimMode::Fixed(3),
            MdsDimModeUi::Planar => MdsDimMode::Visual,
            MdsDimModeUi::Maximum => MdsDimMode::Maximum,
        }
    }

    fn launch_dataset_job(&self, ds: &EtvDataset, output_dir: PathBuf, state: &mut AppState) {
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let all_labels = collect_dataset_labels(ds);
        let (datasets, dropped_edges) = build_dataset_adjacencies(ds, &all_labels);

        let mds_config = self.build_mds_config();
        let dim_mode = self.selected_dim_mode();
        let procrustes_mode = self.procrustes;
        let normalize = self.normalize;
        let target_range = self.norm_range;

        thread::spawn(move || {
            if !dropped_edges.is_empty() {
                eprintln!(
                    "AlignSpace skipped {} edge(s) with unknown labels: {}",
                    dropped_edges.len(),
                    dropped_edges.join(", ")
                );
            }
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

    fn launch_mf_job(
        &self,
        mf_output: &mf_pipeline::types::MfOutput,
        output_dir: PathBuf,
        state: &mut AppState,
    ) {
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let output = mf_output.clone();
        let mds_config = self.build_mds_config();
        let dim_mode = self.selected_dim_mode();
        let procrustes_mode = self.procrustes;
        let normalize = self.normalize;
        let target_range = self.norm_range;

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Embedding MatrixForge output…".into(),
                pct: 0.2,
            });

            let input = build_mf_distance_input(
                &output,
                mds_config,
                procrustes_mode,
                dim_mode,
                normalize,
                target_range,
            );

            match run_distance_pipeline(&input) {
                Ok(result) => {
                    let out_file = output_dir.join("as_result.json");
                    match serde_json::to_string(&result.etv_dataset)
                        .map_err(|e| e.to_string())
                        .and_then(|json| std::fs::write(&out_file, json).map_err(|e| e.to_string()))
                    {
                        Ok(()) => {
                            let _ = tx.send(PipelineEvent::Done(Ok(out_file)));
                        }
                        Err(e) => {
                            let _ = tx.send(PipelineEvent::Done(Err(e)));
                        }
                    }
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

    fn launch_mf_series_job(
        &self,
        series: &mf_pipeline::types::MfSeriesOutput,
        output_dir: PathBuf,
        state: &mut AppState,
    ) {
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let output = series.clone();
        let mds_config = self.build_mds_config();
        let dim_mode = self.selected_dim_mode();
        let procrustes_mode = self.procrustes;
        let normalize = self.normalize;
        let target_range = self.norm_range;

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Embedding MatrixForge series…".into(),
                pct: 0.2,
            });

            let input = mf_series_output_to_as_input(
                &output,
                mds_config,
                procrustes_mode,
                dim_mode,
                normalize,
                target_range,
                true,
            );

            match run_distance_pipeline(&input) {
                Ok(result) => {
                    let out_file = output_dir.join("as_result.json");
                    match serde_json::to_string(&result.etv_dataset)
                        .map_err(|e| e.to_string())
                        .and_then(|json| std::fs::write(&out_file, json).map_err(|e| e.to_string()))
                    {
                        Ok(()) => {
                            let _ = tx.send(PipelineEvent::Done(Ok(out_file)));
                        }
                        Err(e) => {
                            let _ = tx.send(PipelineEvent::Done(Err(e)));
                        }
                    }
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

fn build_mf_distance_input(
    mf_output: &mf_pipeline::types::MfOutput,
    mds_config: MdsConfig,
    procrustes_mode: ProcrustesMode,
    mds_dims: MdsDimMode,
    normalize: bool,
    target_range: f64,
) -> AsDistancePipelineInput {
    let se = mf_output_to_distance_matrix(
        mf_output.labels.clone(),
        &mf_output.similarity_matrix,
        mf_output.n,
        mf_output.sim_to_dist,
    );

    AsDistancePipelineInput {
        datasets: vec![("MatrixForge".to_string(), se)],
        mds_config,
        procrustes_mode,
        mds_dims,
        normalize,
        target_range,
        procrustes_scale: true,
    }
}

fn collect_dataset_labels(ds: &EtvDataset) -> Vec<String> {
    let mut labels = Vec::new();

    for label in &ds.all_labels {
        if !labels.contains(label) {
            labels.push(label.clone());
        }
    }

    for sheet in &ds.sheets {
        for row in &sheet.rows {
            if !labels.contains(&row.label) {
                labels.push(row.label.clone());
            }
        }
    }

    labels
}

fn build_dataset_adjacencies(
    ds: &EtvDataset,
    all_labels: &[String],
) -> (Vec<(String, Array2<f64>)>, Vec<String>) {
    let n = all_labels.len();
    let label_index: HashMap<&str, usize> = all_labels
        .iter()
        .enumerate()
        .map(|(idx, label)| (label.as_str(), idx))
        .collect();
    let mut dropped_edges = BTreeSet::new();

    let datasets = ds
        .sheets
        .iter()
        .map(|sheet| {
            let mut mat = Array2::<f64>::zeros((n, n));
            for edge in &sheet.edges {
                let i = label_index.get(edge.from.as_str()).copied();
                let j = label_index.get(edge.to.as_str()).copied();
                match (i, j) {
                    (Some(i), Some(j)) => {
                        mat[[i, j]] = edge.strength;
                        mat[[j, i]] = edge.strength;
                    }
                    _ => {
                        dropped_edges
                            .insert(format!("{}:{} -> {}", sheet.name, edge.from, edge.to));
                    }
                }
            }
            (sheet.name.clone(), mat)
        })
        .collect();

    (datasets, dropped_edges.into_iter().collect())
}

fn load_etv_dataset(path: &std::path::Path) -> Result<EtvDataset, String> {
    let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::{build_dataset_adjacencies, build_mf_distance_input, collect_dataset_labels};
    use as_pipeline::types::CentralityReport;
    use as_pipeline::types::{MdsConfig, MdsDimMode, ProcrustesMode};
    use lv_data::schema::{EdgeRow, EtvDataset, EtvRow, EtvSheet};
    use mf_pipeline::types::{MfOutput, SimToDistMethod};

    #[test]
    fn collect_dataset_labels_uses_union_across_dataset() {
        let dataset = EtvDataset {
            source_path: None,
            sheets: vec![
                EtvSheet {
                    name: "T1".into(),
                    sheet_index: 0,
                    rows: vec![EtvRow {
                        label: "alpha".into(),
                        ..Default::default()
                    }],
                    edges: vec![],
                },
                EtvSheet {
                    name: "T2".into(),
                    sheet_index: 1,
                    rows: vec![EtvRow {
                        label: "beta".into(),
                        ..Default::default()
                    }],
                    edges: vec![],
                },
            ],
            all_labels: vec!["alpha".into()],
        };

        assert_eq!(
            collect_dataset_labels(&dataset),
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[test]
    fn build_dataset_adjacencies_reports_unknown_edge_labels() {
        let dataset = EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "T1".into(),
                sheet_index: 0,
                rows: vec![EtvRow {
                    label: "alpha".into(),
                    ..Default::default()
                }],
                edges: vec![EdgeRow {
                    from: "alpha".into(),
                    to: "missing".into(),
                    strength: 1.0,
                }],
            }],
            all_labels: vec!["alpha".into()],
        };

        let (datasets, dropped) = build_dataset_adjacencies(&dataset, &["alpha".into()]);

        assert_eq!(datasets.len(), 1);
        assert_eq!(dropped, vec!["T1:alpha -> missing"]);
    }

    #[test]
    fn build_mf_distance_input_creates_single_matrixforge_dataset() {
        let mf_output = MfOutput {
            labels: vec!["alpha".into(), "beta".into()],
            similarity_matrix: vec![1.0, 0.25, 0.25, 1.0],
            sim_to_dist: SimToDistMethod::Linear,
            nppmi_matrix: vec![1.0, 0.25, 0.25, 1.0],
            raw_counts: vec![0; 4],
            ppmi_matrix: vec![0.0; 4],
            n: 2,
            centrality: CentralityReport {
                labels: vec!["alpha".into(), "beta".into()],
                degree: vec![0.0; 2],
                distance: vec![0.0; 2],
                closeness: vec![0.0; 2],
                betweenness: vec![0.0; 2],
            },
        };

        let input = build_mf_distance_input(
            &mf_output,
            MdsConfig::Classical,
            ProcrustesMode::None,
            MdsDimMode::Fixed(3),
            true,
            300.0,
        );

        assert_eq!(input.datasets.len(), 1);
        assert_eq!(input.datasets[0].0, "MatrixForge");
        assert_eq!(input.datasets[0].1.labels, mf_output.labels);
        assert_eq!(input.mds_dims, MdsDimMode::Fixed(3));
        assert!(input.normalize);
    }
}
