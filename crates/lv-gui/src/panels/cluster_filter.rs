//! Cluster filter panel.

use crate::state::{AppState, EgoEdgeDirection};

/// Panel for filtering objects by cluster value.
#[derive(Default)]
pub struct ClusterFilterPanel;

impl ClusterFilterPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.heading("Cluster Filter");
        ui.separator();

        // Determine the actual data range
        let (data_min, data_max) = if let Some(ds) = &state.dataset {
            let vals: Vec<f64> = ds
                .sheets
                .iter()
                .flat_map(|s| s.rows.iter().map(|r| r.cluster_value))
                .collect();
            let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (mn.min(0.0), mx.max(1.0))
        } else {
            (0.0, 1.0)
        };

        // Clamp stored range to data range
        state.cluster_min = state.cluster_min.clamp(data_min, data_max);
        state.cluster_max = state.cluster_max.clamp(data_min, data_max);
        if state.cluster_min > state.cluster_max {
            state.cluster_min = data_min;
            state.cluster_max = data_max;
        }

        ui.horizontal(|ui| {
            ui.label("Min:");
            ui.add(
                egui::DragValue::new(&mut state.cluster_min)
                    .speed(0.01)
                    .range(data_min..=state.cluster_max),
            );
            ui.label("Max:");
            ui.add(
                egui::DragValue::new(&mut state.cluster_max)
                    .speed(0.01)
                    .range(state.cluster_min..=data_max),
            );
        });

        if ui.button("Reset Range").clicked() {
            state.cluster_min = data_min;
            state.cluster_max = data_max;
        }

        ui.separator();
        ui.checkbox(&mut state.ego_mode, "Show ego cluster");
        ui.horizontal(|ui| {
            ui.label("Edge direction:");
            egui::ComboBox::from_id_salt("ego_direction")
                .selected_text(match state.ego_direction {
                    EgoEdgeDirection::Incoming => "Incoming",
                    EgoEdgeDirection::Outgoing => "Outgoing",
                    EgoEdgeDirection::Both => "Both",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.ego_direction,
                        EgoEdgeDirection::Incoming,
                        "Incoming",
                    );
                    ui.selectable_value(
                        &mut state.ego_direction,
                        EgoEdgeDirection::Outgoing,
                        "Outgoing",
                    );
                    ui.selectable_value(&mut state.ego_direction, EgoEdgeDirection::Both, "Both");
                });
        });
        ui.checkbox(&mut state.secondary_edges, "Include secondary edges");
        ui.checkbox(&mut state.shared_only, "Shared objects only");

        // Object list (filtered)
        if let Some(ds) = &state.dataset {
            let visible: Vec<&str> = ds
                .sheets
                .iter()
                .flat_map(|s| s.rows.iter())
                .filter(|r| {
                    r.cluster_value >= state.cluster_min && r.cluster_value <= state.cluster_max
                })
                .map(|r| r.label.as_str())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            ui.label(format!("{} objects visible", visible.len()));

            egui::ScrollArea::vertical()
                .max_height(150.0)
                .show(ui, |ui| {
                    for label in &visible {
                        ui.label(*label);
                    }
                });
        }
    }
}
