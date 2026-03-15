//! Per-object shape / colour / size override panel.

use std::collections::HashMap;

use lv_data::ShapeKind;

use crate::state::{AppState, ObjectOverride};

/// Panel that lets the user override shape, colour, and size for individual objects.
#[derive(Default)]
pub struct ShapeOverridePanel {
    search: String,
}

impl ShapeOverridePanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.heading("Shape Overrides");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.search);
            if ui.button("Reset All").clicked() {
                state.overrides.clear();
            }
        });

        let labels: Vec<String> = if let Some(ds) = &state.dataset {
            ds.all_labels
                .iter()
                .filter(|l| {
                    self.search.is_empty() || l.to_lowercase().contains(&self.search.to_lowercase())
                })
                .cloned()
                .collect()
        } else {
            vec![]
        };

        if labels.is_empty() {
            ui.label("No objects.");
            return;
        }

        egui::ScrollArea::vertical()
            .max_height(300.0)
            .show(ui, |ui| {
                egui::Grid::new("shape_override_grid")
                    .num_columns(5)
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Label");
                        ui.strong("Shape");
                        ui.strong("Color");
                        ui.strong("Size");
                        ui.strong("Reset");
                        ui.end_row();

                        let mut to_clear: Vec<String> = Vec::new();
                        let mut pending: HashMap<String, ObjectOverride> = HashMap::new();

                        for label in &labels {
                            let ov = state.overrides.entry(label.clone()).or_insert_with(|| {
                                ObjectOverride {
                                    shape: None,
                                    color: None,
                                    size: None,
                                }
                            });

                            ui.label(label);

                            // Shape combo
                            let shape_label = ov
                                .shape
                                .map(|s| format!("{s:?}"))
                                .unwrap_or_else(|| "Default".into());
                            egui::ComboBox::from_id_salt(format!("shape_{label}"))
                                .selected_text(&shape_label)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut ov.shape, None, "Default")
                                        .clicked();
                                    for sk in ShapeKind::ALL {
                                        ui.selectable_value(
                                            &mut ov.shape,
                                            Some(sk),
                                            format!("{sk:?}"),
                                        )
                                        .clicked();
                                    }
                                });

                            // Color picker
                            let mut col = ov.color.unwrap_or([0.8, 0.8, 0.8]);
                            if ui.color_edit_button_rgb(&mut col).changed() {
                                ov.color = Some(col);
                            }

                            // Size drag
                            let mut sz = ov.size.unwrap_or(0.0);
                            if ui
                                .add(egui::DragValue::new(&mut sz).speed(0.1).range(0.0..=1000.0))
                                .changed()
                            {
                                ov.size = if sz > 0.0 { Some(sz) } else { None };
                            }

                            // Reset row
                            if ui.small_button("✕").clicked() {
                                to_clear.push(label.clone());
                            }

                            // Collect changes (borrow checker: clone into pending map)
                            pending.insert(label.clone(), ov.clone());

                            ui.end_row();
                        }

                        for key in to_clear {
                            state.overrides.remove(&key);
                        }
                        for (k, v) in pending {
                            state.overrides.insert(k, v);
                        }
                    });
            });
    }
}
