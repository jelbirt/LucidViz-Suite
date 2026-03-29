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
pub use state::{AppState, AudioState, ClusterState, EgoEdgeDirection, ExportState, SessionState};
pub use workspace::{ActiveTab, LucidWorkspace, LvPanels};

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(1 + 1, 2);
    }
}
