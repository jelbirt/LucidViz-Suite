//! Cluster filter panel.

use crate::state::{AppState, EgoEdgeDirection};

/// Panel for filtering objects by cluster value.
#[derive(Default)]
pub struct ClusterFilterPanel;

impl ClusterFilterPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
        ui.heading("Cluster Filter");
        ui.separator();

        // Use cached data range if available, otherwise compute and cache.
        let (data_min, data_max) = if let Some(range) = state.cluster.cached_data_range {
            range
        } else if let Some(ds) = state.dataset() {
            let mut mn = f64::INFINITY;
            let mut mx = f64::NEG_INFINITY;
            for s in &ds.sheets {
                for r in &s.rows {
                    if r.cluster_value < mn {
                        mn = r.cluster_value;
                    }
                    if r.cluster_value > mx {
                        mx = r.cluster_value;
                    }
                }
            }
            let range = (mn.min(0.0), mx.max(1.0));
            state.cluster.cached_data_range = Some(range);
            range
        } else {
            (0.0, 1.0)
        };

        // Clamp stored range to data range
        state.cluster.min = state.cluster.min.clamp(data_min, data_max);
        state.cluster.max = state.cluster.max.clamp(data_min, data_max);
        if state.cluster.min > state.cluster.max {
            state.cluster.min = data_min;
            state.cluster.max = data_max;
        }

        ui.horizontal(|ui| {
            ui.label("Min:");
            ui.add(
                egui::DragValue::new(&mut state.cluster.min)
                    .speed(0.01)
                    .range(data_min..=state.cluster.max),
            );
            ui.label("Max:");
            ui.add(
                egui::DragValue::new(&mut state.cluster.max)
                    .speed(0.01)
                    .range(state.cluster.min..=data_max),
            );
        });

        if ui.button("Reset Range").clicked() {
            state.cluster.min = data_min;
            state.cluster.max = data_max;
        }

        ui.separator();
        ui.checkbox(&mut state.cluster.ego_mode, "Show ego cluster");
        ui.horizontal(|ui| {
            ui.label("Edge direction:");
            egui::ComboBox::from_id_salt("ego_direction")
                .selected_text(match state.cluster.ego_direction {
                    EgoEdgeDirection::Incoming => "Incoming",
                    EgoEdgeDirection::Outgoing => "Outgoing",
                    EgoEdgeDirection::Both => "Both",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut state.cluster.ego_direction,
                        EgoEdgeDirection::Incoming,
                        "Incoming",
                    );
                    ui.selectable_value(
                        &mut state.cluster.ego_direction,
                        EgoEdgeDirection::Outgoing,
                        "Outgoing",
                    );
                    ui.selectable_value(
                        &mut state.cluster.ego_direction,
                        EgoEdgeDirection::Both,
                        "Both",
                    );
                });
        });
        ui.checkbox(
            &mut state.cluster.secondary_edges,
            "Include secondary edges",
        );
        ui.checkbox(&mut state.cluster.shared_only, "Shared objects only");

        // Object list (filtered)
        if let Some(ds) = state.dataset() {
            let visible: Vec<&str> = ds
                .sheets
                .iter()
                .flat_map(|s| s.rows.iter())
                .filter(|r| {
                    r.cluster_value >= state.cluster.min && r.cluster_value <= state.cluster.max
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
