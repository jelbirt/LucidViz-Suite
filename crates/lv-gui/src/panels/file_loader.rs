//! File loader panel — XLSX / JSON file picker and dataset summary.

use lv_data::{load_dataset_json, read_lv_xlsx, LvDataset};

use crate::state::AppState;

/// Events emitted by the file loader panel.
pub enum FileLoaderEvent {
    Loaded {
        dataset: LvDataset,
        path: std::path::PathBuf,
    },
    Error(String),
    None,
}

/// Side-panel for picking and reloading dataset files.
#[derive(Default)]
pub struct FileLoaderPanel;

impl FileLoaderPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) -> FileLoaderEvent {
        ui.heading("Dataset");
        ui.separator();

        let mut ev = FileLoaderEvent::None;

        if ui.button("Open XLSX…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("XLSX", &["xlsx"])
                .pick_file()
            {
                match read_lv_xlsx(&path) {
                    Ok(dataset) => ev = FileLoaderEvent::Loaded { dataset, path },
                    Err(e) => ev = FileLoaderEvent::Error(format!("XLSX load error: {e}")),
                }
            }
        }

        if ui.button("Open JSON…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("JSON", &["json"])
                .pick_file()
            {
                match load_dataset_json(&path) {
                    Ok(dataset) => ev = FileLoaderEvent::Loaded { dataset, path },
                    Err(e) => ev = FileLoaderEvent::Error(format!("JSON load error: {e}")),
                }
            }
        }

        if let Some(path) = state.source_path() {
            let can_reload = path.exists();
            if ui
                .add_enabled(can_reload, egui::Button::new("Reload"))
                .clicked()
            {
                let p = path.clone();
                let result = if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    load_dataset_json(&p).map_err(|e| e.to_string())
                } else {
                    read_lv_xlsx(&p).map_err(|e| e.to_string())
                };
                match result {
                    Ok(dataset) => ev = FileLoaderEvent::Loaded { dataset, path: p },
                    Err(e) => ev = FileLoaderEvent::Error(e),
                }
            }
        }

        // ── Recent files ──────────────────────────────────────────────
        if !state.recent_files.is_empty() {
            ui.separator();
            ui.label("Recent files:");
            for path in state.recent_files.clone() {
                let label = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("<unknown>");
                if ui
                    .small_button(label)
                    .on_hover_text(path.display().to_string())
                    .clicked()
                {
                    let result = if path.extension().and_then(|e| e.to_str()) == Some("json") {
                        load_dataset_json(&path).map_err(|e| e.to_string())
                    } else {
                        read_lv_xlsx(&path).map_err(|e| e.to_string())
                    };
                    match result {
                        Ok(dataset) => ev = FileLoaderEvent::Loaded { dataset, path },
                        Err(e) => ev = FileLoaderEvent::Error(e),
                    }
                }
            }
        }

        ui.separator();

        // Dataset summary
        if let Some(ds) = state.dataset() {
            ui.label(format!(
                "File: {}",
                state
                    .source_path()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .unwrap_or("<unknown>")
            ));
            ui.label(format!("Time points: {}", ds.time_points()));
            ui.label(format!("Max objects: {}", ds.max_objects()));

            // ── Dataset preview ───────────────────────────────────────
            egui::CollapsingHeader::new("Preview")
                .default_open(false)
                .show(ui, |ui| {
                    // First 10 labels
                    let labels: Vec<&str> =
                        ds.all_labels.iter().take(10).map(|s| s.as_str()).collect();
                    ui.label(format!(
                        "Labels (first {}): {}",
                        labels.len(),
                        labels.join(", ")
                    ));

                    // Edge count
                    let total_edges: usize = ds.sheets.iter().map(|s| s.edges.len()).sum();
                    ui.label(format!("Total edges: {total_edges}"));

                    // Cluster value range
                    let mut cv_min = f64::INFINITY;
                    let mut cv_max = f64::NEG_INFINITY;
                    for sheet in &ds.sheets {
                        for row in &sheet.rows {
                            cv_min = cv_min.min(row.cluster_value);
                            cv_max = cv_max.max(row.cluster_value);
                        }
                    }
                    if cv_min.is_finite() && cv_max.is_finite() {
                        ui.label(format!("Cluster range: [{cv_min:.2}, {cv_max:.2}]"));
                    }

                    // Shape distribution
                    let mut shape_counts = std::collections::HashMap::new();
                    for sheet in &ds.sheets {
                        for row in &sheet.rows {
                            *shape_counts
                                .entry(format!("{:?}", row.shape))
                                .or_insert(0u32) += 1;
                        }
                    }
                    let mut shapes: Vec<(String, u32)> = shape_counts.into_iter().collect();
                    shapes.sort_by(|a, b| b.1.cmp(&a.1));
                    let shape_str: Vec<String> =
                        shapes.iter().map(|(s, c)| format!("{s}: {c}")).collect();
                    ui.label(format!("Shapes: {}", shape_str.join(", ")));
                });
        } else {
            ui.label("No dataset loaded.");
        }

        if let Some(err) = &state.load_error {
            ui.colored_label(egui::Color32::RED, err);
        }

        ev
    }
}
