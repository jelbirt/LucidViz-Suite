//! `lv-gui` — egui immediate-mode panels for the Lucid Visualization Suite.
//!
//! See `implementation_plan.md` §5 for full specification.

pub mod as_panel;
pub mod bridge;
pub mod mf_panel;
pub mod panels;
pub mod state;
pub mod workspace;

pub use as_panel::AsPanel;
pub use mf_panel::MfPanel;

/// Tooltip discoverability marker color: muted blue.
const HELP_COLOR: egui::Color32 = egui::Color32::from_rgb(120, 160, 220);

/// Draw a small "(?) " marker that shows a tooltip on hover.
pub fn help_marker(ui: &mut egui::Ui, tooltip: &str) {
    ui.colored_label(HELP_COLOR, "(?)").on_hover_text(tooltip);
}
pub use state::{
    AppState, AppStateSnapshot, AudioState, ClusterState, EgoEdgeDirection, ExportState,
    SessionState, UndoStack,
};
pub use workspace::{ActiveTab, LucidWorkspace, LvPanels};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(1 + 1, 2);
    }
}
