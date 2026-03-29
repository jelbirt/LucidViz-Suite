//! AlignSpace pipeline panel.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use crate::bridge::{mf_series_output_to_as_input, MfSeriesAsInputOptions};
use as_pipeline::output::write_as_results;
use as_pipeline::pipeline::{mf_output_to_distance_matrix, run_distance_pipeline, run_pipeline};
use as_pipeline::types::{
    AsDistancePipelineInput, AsPipelineInput, AsPipelineResult, CentralityMode, MdsConfig,
    MdsDimMode, NormalizationMode, ProcrustesMode, SmacofConfig, SmacofInit,
};
use lv_data::{write_lv_json, LvDataset};
use ndarray::Array2;

use crate::state::{
    AppState, AsInputSource, AsRunOutcome, PipelineEvent, PipelineJob, PipelineOutcome,
};

/// Panel for configuring and launching the AlignSpace MDS pipeline.
pub struct AsPanel {
    // MDS config
    algorithm: MdsAlgorithmUi,
    dim_mode: MdsDimModeUi,
    procrustes: ProcrustesMode,
    normalize: bool,
    normalization_mode: NormalizationMode,
    norm_range: f64,
    centrality_mode: CentralityMode,
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
            normalization_mode: NormalizationMode::Independent,
            norm_range: 300.0,
            centrality_mode: CentralityMode::Directed,
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
                    ProcrustesMode::TimeSeriesAnchored,
                    "Time Series (Anchored)",
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
            egui::ComboBox::from_label("Normalization scope")
                .selected_text(match self.normalization_mode {
                    NormalizationMode::Independent => "Per slice",
                    NormalizationMode::Global => "Whole series",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.normalization_mode,
                        NormalizationMode::Independent,
                        "Per slice",
                    );
                    ui.selectable_value(
                        &mut self.normalization_mode,
                        NormalizationMode::Global,
                        "Whole series",
                    );
                });
            ui.horizontal(|ui| {
                ui.label("Range:");
                ui.add(
                    egui::DragValue::new(&mut self.norm_range)
                        .speed(1.0)
                        .range(1.0..=10000.0),
                );
            });
        }

        egui::ComboBox::from_label("Centrality")
            .selected_text(match self.centrality_mode {
                CentralityMode::Directed => "Directed",
                CentralityMode::UndirectedLegacy => "Undirected (legacy)",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(
                    &mut self.centrality_mode,
                    CentralityMode::Directed,
                    "Directed",
                );
                ui.selectable_value(
                    &mut self.centrality_mode,
                    CentralityMode::UndirectedLegacy,
                    "Undirected (legacy)",
                );
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
        let has_input = match state.as_input_source {
            AsInputSource::Dataset => state.dataset().is_some(),
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
                    if let Some(ds) = state.dataset().cloned() {
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
                    Ok(PipelineOutcome::AsRun(outcome)) => {
                        self.status = format!("Done: {}", outcome.dataset_path.display());
                        if !outcome.warnings.is_empty() {
                            ui.separator();
                            ui.label(format!("Warnings: {}", outcome.warnings.join(" | ")));
                        }
                        if ui.button("Load output into LV").clicked() {
                            state.queue_dataset_load(
                                outcome.dataset.clone(),
                                outcome.dataset_path.clone(),
                            );
                            self.status = "AlignSpace output loaded into LV.".into();
                        }
                    }
                    Ok(PipelineOutcome::File(_)) => {
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

    fn launch_dataset_job(&self, ds: &LvDataset, output_dir: PathBuf, state: &mut AppState) {
        let (tx, rx) = mpsc::channel::<PipelineEvent>();

        let source_dataset = ds.clone();
        let all_labels = collect_dataset_labels(ds);
        let (datasets, dropped_edges) = build_dataset_adjacencies(ds, &all_labels);

        let mds_config = self.build_mds_config();
        let dim_mode = self.selected_dim_mode();
        let procrustes_mode = self.procrustes;
        let normalize = self.normalize;
        let normalization_mode = self.normalization_mode;
        let target_range = self.norm_range;
        let centrality_mode = self.centrality_mode;

        thread::spawn(move || {
            let warnings = dropped_edge_warnings(&dropped_edges);
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
                normalization_mode,
                target_range,
                procrustes_scale: true,
                centrality_mode,
            };
            match run_pipeline(&input) {
                Ok(mut result) => {
                    preserve_lv_metadata(&mut result, &source_dataset);
                    let _ = tx.send(PipelineEvent::Progress {
                        step: "Writing output…".into(),
                        pct: 0.9,
                    });
                    let result = write_gui_output_artifacts(&result, &output_dir, &warnings);
                    let _ = tx.send(PipelineEvent::Done(result.map(PipelineOutcome::AsRun)));
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
        let normalization_mode = self.normalization_mode;
        let target_range = self.norm_range;
        let centrality_mode = self.centrality_mode;

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Embedding MatrixForge output…".into(),
                pct: 0.2,
            });

            let input = match build_mf_distance_input(
                &output,
                MfDistanceInputSettings {
                    mds_config,
                    procrustes_mode,
                    mds_dims: dim_mode,
                    normalize,
                    normalization_mode,
                    target_range,
                    centrality_mode,
                },
            ) {
                Ok(input) => input,
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Done(Err(e.to_string())));
                    return;
                }
            };

            match run_distance_pipeline(&input) {
                Ok(result) => {
                    let result = write_gui_output_artifacts(&result, &output_dir, &[]);
                    let _ = tx.send(PipelineEvent::Done(result.map(PipelineOutcome::AsRun)));
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
        let normalization_mode = self.normalization_mode;
        let target_range = self.norm_range;
        let centrality_mode = self.centrality_mode;

        thread::spawn(move || {
            let _ = tx.send(PipelineEvent::Progress {
                step: "Embedding MatrixForge series…".into(),
                pct: 0.2,
            });

            let input = match mf_series_output_to_as_input(
                &output,
                MfSeriesAsInputOptions {
                    mds_config,
                    procrustes_mode,
                    mds_dims: dim_mode,
                    normalize,
                    normalization_mode,
                    target_range,
                    procrustes_scale: true,
                    centrality_mode,
                },
            ) {
                Ok(input) => input,
                Err(e) => {
                    let _ = tx.send(PipelineEvent::Done(Err(e.to_string())));
                    return;
                }
            };

            match run_distance_pipeline(&input) {
                Ok(result) => {
                    let result = write_gui_output_artifacts(&result, &output_dir, &[]);
                    let _ = tx.send(PipelineEvent::Done(result.map(PipelineOutcome::AsRun)));
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

fn preserve_lv_metadata(result: &mut AsPipelineResult, source: &LvDataset) {
    for (sheet_idx, output_sheet) in result.lv_dataset.sheets.iter_mut().enumerate() {
        let Some(source_sheet) = source.sheets.get(sheet_idx) else {
            continue;
        };
        let source_rows: HashMap<&str, &lv_data::schema::LvRow> = source_sheet
            .rows
            .iter()
            .map(|row| (row.label.as_str(), row))
            .collect();

        for output_row in &mut output_sheet.rows {
            let Some(source_row) = source_rows.get(output_row.label.as_str()) else {
                continue;
            };
            let (x, y, z) = (output_row.x, output_row.y, output_row.z);
            let mut preserved = (*source_row).clone();
            preserved.x = x;
            preserved.y = y;
            preserved.z = z;
            *output_row = preserved;
        }
    }

    result.lv_dataset.all_labels =
        LvDataset::canonical_all_labels_from_sheets(&result.lv_dataset.sheets);
}

#[derive(Debug, Clone)]
struct MfDistanceInputSettings {
    mds_config: MdsConfig,
    procrustes_mode: ProcrustesMode,
    mds_dims: MdsDimMode,
    normalize: bool,
    normalization_mode: NormalizationMode,
    target_range: f64,
    centrality_mode: CentralityMode,
}

fn build_mf_distance_input(
    mf_output: &mf_pipeline::types::MfOutput,
    settings: MfDistanceInputSettings,
) -> Result<AsDistancePipelineInput, String> {
    mf_output
        .validate()
        .map_err(|e| format!("Invalid MatrixForge output: {e}"))?;
    let se = mf_output_to_distance_matrix(
        mf_output.labels.clone(),
        &mf_output.similarity_matrix,
        mf_output.n,
        mf_output.sim_to_dist,
    )
    .map_err(|e| e.to_string())?;

    Ok(AsDistancePipelineInput {
        datasets: vec![("MatrixForge".to_string(), se)],
        mds_config: settings.mds_config,
        procrustes_mode: settings.procrustes_mode,
        mds_dims: settings.mds_dims,
        normalize: settings.normalize,
        normalization_mode: settings.normalization_mode,
        target_range: settings.target_range,
        procrustes_scale: true,
        centrality_mode: settings.centrality_mode,
    })
}

fn collect_dataset_labels(ds: &LvDataset) -> Vec<String> {
    ds.canonical_all_labels()
}

fn build_dataset_adjacencies(
    ds: &LvDataset,
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

fn dropped_edge_warnings(dropped_edges: &[String]) -> Vec<String> {
    if dropped_edges.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "AlignSpace skipped {} edge(s) with unknown labels: {}",
            dropped_edges.len(),
            dropped_edges.join(", ")
        )]
    }
}

fn write_warnings_file(output_dir: &std::path::Path, warnings: &[String]) -> Result<(), String> {
    if warnings.is_empty() {
        return Ok(());
    }
    std::fs::create_dir_all(output_dir).map_err(|e| e.to_string())?;
    std::fs::write(output_dir.join("warnings.txt"), warnings.join("\n")).map_err(|e| e.to_string())
}

fn write_gui_output_artifacts(
    result: &AsPipelineResult,
    output_dir: &std::path::Path,
    warnings: &[String],
) -> Result<AsRunOutcome, String> {
    write_as_results(result, output_dir).map_err(|e| e.to_string())?;

    let dataset_path = output_dir.join("lv_dataset.json");
    write_lv_json(&result.lv_dataset, &dataset_path).map_err(|e| e.to_string())?;
    write_warnings_file(output_dir, warnings)?;

    Ok(AsRunOutcome {
        dataset: result.lv_dataset.clone(),
        dataset_path,
        warnings: warnings.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_dataset_adjacencies, build_mf_distance_input, collect_dataset_labels,
        preserve_lv_metadata, write_gui_output_artifacts,
    };
    use as_pipeline::types::CentralityReport;
    use as_pipeline::types::{
        AsPipelineResult, CentralityMode, CentralityState, DistanceMatrix, MdsAlgorithm,
        MdsCoordinates, ProcrustesResult,
    };
    use as_pipeline::types::{MdsConfig, MdsDimMode, NormalizationMode, ProcrustesMode};
    use lv_data::schema::{EdgeRow, LvDataset, LvRow, LvSheet};
    use mf_pipeline::types::{MfOutput, SimToDistMethod};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn collect_dataset_labels_uses_union_across_dataset() {
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![
                LvSheet {
                    name: "T1".into(),
                    sheet_index: 0,
                    rows: vec![LvRow {
                        label: "alpha".into(),
                        ..Default::default()
                    }],
                    edges: vec![],
                },
                LvSheet {
                    name: "T2".into(),
                    sheet_index: 1,
                    rows: vec![LvRow {
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
    fn collect_dataset_labels_discards_stale_all_labels_entries() {
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "T1".into(),
                sheet_index: 0,
                rows: vec![LvRow {
                    label: "alpha".into(),
                    ..Default::default()
                }],
                edges: vec![],
            }],
            all_labels: vec!["ghost".into(), "alpha".into()],
        };

        assert_eq!(collect_dataset_labels(&dataset), vec!["alpha".to_string()]);
    }

    #[test]
    fn build_dataset_adjacencies_reports_unknown_edge_labels() {
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "T1".into(),
                sheet_index: 0,
                rows: vec![LvRow {
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
    fn build_dataset_adjacencies_preserves_edge_direction() {
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "T1".into(),
                sheet_index: 0,
                rows: vec![
                    LvRow {
                        label: "alpha".into(),
                        ..Default::default()
                    },
                    LvRow {
                        label: "beta".into(),
                        ..Default::default()
                    },
                ],
                edges: vec![EdgeRow {
                    from: "alpha".into(),
                    to: "beta".into(),
                    strength: 0.75,
                }],
            }],
            all_labels: vec!["alpha".into(), "beta".into()],
        };

        let (datasets, dropped) =
            build_dataset_adjacencies(&dataset, &["alpha".into(), "beta".into()]);

        assert!(dropped.is_empty());
        assert_eq!(datasets.len(), 1);
        let matrix = &datasets[0].1;
        assert!((matrix[[0, 1]] - 0.75).abs() < 1e-12);
        assert_eq!(matrix[[1, 0]], 0.0);
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
            super::MfDistanceInputSettings {
                mds_config: MdsConfig::Classical,
                procrustes_mode: ProcrustesMode::None,
                mds_dims: MdsDimMode::Fixed(3),
                normalize: true,
                normalization_mode: NormalizationMode::Independent,
                target_range: 300.0,
                centrality_mode: CentralityMode::Directed,
            },
        )
        .expect("valid MF output should convert");

        assert_eq!(input.datasets.len(), 1);
        assert_eq!(input.datasets[0].0, "MatrixForge");
        assert_eq!(input.datasets[0].1.labels, mf_output.labels);
        assert_eq!(input.mds_dims, MdsDimMode::Fixed(3));
        assert!(input.normalize);
    }

    #[test]
    fn write_gui_output_artifacts_preserves_core_as_json_contract() {
        let result = AsPipelineResult {
            coordinates: vec![MdsCoordinates::new(
                vec!["alpha".into()],
                vec![1.0, 2.0, 3.0],
                3,
                0.0,
                MdsAlgorithm::Classical,
            )
            .expect("test coordinates should build")],
            procrustes: vec![ProcrustesResult {
                aligned: MdsCoordinates::new(
                    vec!["alpha".into()],
                    vec![1.0, 2.0, 3.0],
                    3,
                    0.0,
                    MdsAlgorithm::Classical,
                )
                .expect("test coordinates should build"),
                rotation: vec![1.0, 0.0, 0.0, 1.0],
                scale: 1.0,
                translation: vec![0.0, 0.0],
                residual: 0.0,
            }],
            centralities: vec![CentralityState::Unavailable {
                labels: vec!["alpha".into()],
                reason: "distance-only".into(),
            }],
            centrality_mode: CentralityMode::Directed,
            distance_matrices: vec![DistanceMatrix::new(vec!["alpha".into()], vec![0.0])
                .expect("test distance matrix should build")],
            lv_dataset: LvDataset {
                source_path: None,
                sheets: vec![LvSheet {
                    name: "T1".into(),
                    sheet_index: 0,
                    rows: vec![LvRow {
                        label: "alpha".into(),
                        ..Default::default()
                    }],
                    edges: vec![],
                }],
                all_labels: vec!["alpha".into()],
            },
        };

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time failed")
            .as_nanos();
        let out_dir = std::env::temp_dir().join(format!("as-panel-artifacts-{stamp}"));

        let outcome =
            write_gui_output_artifacts(&result, &out_dir, &[]).expect("artifacts should write");

        assert_eq!(
            outcome.dataset_path.file_name().and_then(|s| s.to_str()),
            Some("lv_dataset.json")
        );

        let as_result_json = std::fs::read_to_string(out_dir.join("as_result.json"))
            .expect("as_result.json should exist");
        assert!(as_result_json.contains("\"procrustes\""));
        assert!(as_result_json.contains("\"lv_dataset\""));

        let dataset_json =
            std::fs::read_to_string(&outcome.dataset_path).expect("lv dataset json should exist");
        assert!(dataset_json.contains("\"all_labels\""));
        assert!(!dataset_json.contains("\"procrustes\""));

        let _ = std::fs::remove_dir_all(out_dir);
    }

    #[test]
    fn build_mf_distance_input_rejects_invalid_shape() {
        let mf_output = MfOutput {
            labels: vec!["alpha".into(), "beta".into()],
            similarity_matrix: vec![1.0, 0.25, 0.25],
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

        let err = build_mf_distance_input(
            &mf_output,
            super::MfDistanceInputSettings {
                mds_config: MdsConfig::Classical,
                procrustes_mode: ProcrustesMode::None,
                mds_dims: MdsDimMode::Fixed(3),
                normalize: true,
                normalization_mode: NormalizationMode::Independent,
                target_range: 300.0,
                centrality_mode: CentralityMode::Directed,
            },
        )
        .expect_err("invalid MF output must fail");

        assert!(err.contains("similarity_matrix expected 4 values"));
    }

    #[test]
    fn preserve_lv_metadata_keeps_source_row_fields_and_updates_positions() {
        let source = LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "T1".into(),
                sheet_index: 0,
                rows: vec![LvRow {
                    label: "alpha".into(),
                    x: 10.0,
                    y: 20.0,
                    z: 30.0,
                    size: 2.5,
                    size_alpha: 0.75,
                    spin_x: 1.0,
                    spin_y: 2.0,
                    spin_z: 3.0,
                    shape: lv_data::ShapeKind::Cube,
                    color_r: 0.2,
                    color_g: 0.3,
                    color_b: 0.4,
                    note: 72,
                    instrument: 33,
                    channel: 4,
                    velocity: 99,
                    cluster_value: 7.0,
                    beats: 3,
                }],
                edges: vec![],
            }],
            all_labels: vec!["alpha".into()],
        };

        let mut result = AsPipelineResult {
            coordinates: vec![],
            procrustes: vec![],
            centralities: vec![],
            centrality_mode: CentralityMode::Directed,
            distance_matrices: vec![],
            lv_dataset: LvDataset {
                source_path: None,
                sheets: vec![LvSheet {
                    name: "T1".into(),
                    sheet_index: 0,
                    rows: vec![LvRow {
                        label: "alpha".into(),
                        x: 1.0,
                        y: 2.0,
                        z: 3.0,
                        ..Default::default()
                    }],
                    edges: vec![],
                }],
                all_labels: vec!["alpha".into()],
            },
        };

        preserve_lv_metadata(&mut result, &source);

        let row = &result.lv_dataset.sheets[0].rows[0];
        assert_eq!(row.x, 1.0);
        assert_eq!(row.y, 2.0);
        assert_eq!(row.z, 3.0);
        assert_eq!(row.shape, lv_data::ShapeKind::Cube);
        assert_eq!(row.size, 2.5);
        assert_eq!(row.size_alpha, 0.75);
        assert_eq!(row.spin_y, 2.0);
        assert_eq!(row.note, 72);
        assert_eq!(row.instrument, 33);
        assert_eq!(row.channel, 4);
        assert_eq!(row.velocity, 99);
        assert_eq!(row.cluster_value, 7.0);
        assert_eq!(row.beats, 3);
        assert!((row.color_r - 0.2).abs() < f32::EPSILON);
        assert!((row.color_g - 0.3).abs() < f32::EPSILON);
        assert!((row.color_b - 0.4).abs() < f32::EPSILON);
    }
}
