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
        } else {
            ui.label("No dataset loaded.");
        }

        if let Some(err) = &state.load_error {
            ui.colored_label(egui::Color32::RED, err);
        }

        ev
    }
}
