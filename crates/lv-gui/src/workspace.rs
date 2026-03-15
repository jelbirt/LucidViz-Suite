use egui::{CentralPanel, Context, TopBottomPanel};

use crate::as_panel::AsPanel;
use crate::mf_panel::MfPanel;
use crate::panels::{
    AudioPanel, ClusterFilterPanel, ExportPanel, FileLoaderEvent, FileLoaderPanel, LisControlPanel,
    LisEvent, ShapeOverridePanel,
};
use crate::state::AppState;

// ────────────────────────────────────────────────────────────────────────────
// Tab enum
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActiveTab {
    #[default]
    LV,
    AlignSpace,
    MatrixForge,
}

// ────────────────────────────────────────────────────────────────────────────
// Panel collection for the LV (main) tab
// ────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct LvPanels {
    pub file_loader: FileLoaderPanel,
    pub lis_controls: LisControlPanel,
    pub shape_overrides: ShapeOverridePanel,
    pub cluster_filter: ClusterFilterPanel,
    pub audio_panel: AudioPanel,
    pub export_panel: ExportPanel,
}

// ────────────────────────────────────────────────────────────────────────────
// Top-level workspace
// ────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct LucidWorkspace {
    pub active_tab: ActiveTab,
    pub lv_panels: LvPanels,
    pub as_panel: AsPanel,
    pub mf_panel: MfPanel,
}

impl LucidWorkspace {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the full GUI for one frame.
    ///
    /// Returns `true` if the renderer should rebuild the `LisBuffer` (i.e. the
    /// dataset or LIS config changed).
    pub fn show(&mut self, ctx: &Context, state: &mut AppState) -> bool {
        let mut needs_rebuild = false;

        // ── Top bar: tab selector ───────────────────────────────────────────
        TopBottomPanel::top("tab_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, ActiveTab::LV, "LV");
                ui.selectable_value(&mut self.active_tab, ActiveTab::AlignSpace, "AlignSpace");
                ui.selectable_value(&mut self.active_tab, ActiveTab::MatrixForge, "MatrixForge");
            });
        });

        // ── Main panel: tab contents ────────────────────────────────────────
        CentralPanel::default().show(ctx, |ui| match self.active_tab {
            ActiveTab::LV => {
                needs_rebuild |= self.show_lv_tab(ui, state);
            }
            ActiveTab::AlignSpace => {
                self.as_panel.show(ui, state);
            }
            ActiveTab::MatrixForge => {
                self.mf_panel.show(ui, state);
            }
        });

        needs_rebuild
    }

    // ── LV tab layout ───────────────────────────────────────────────────────

    fn show_lv_tab(&mut self, ui: &mut egui::Ui, state: &mut AppState) -> bool {
        let mut needs_rebuild = false;

        // File loader — top section
        egui::CollapsingHeader::new("File")
            .default_open(true)
            .show(ui, |ui| {
                let evt = self.lv_panels.file_loader.show(ui, state);
                if matches!(evt, FileLoaderEvent::Loaded(_)) {
                    needs_rebuild = true;
                }
            });

        ui.separator();

        // LIS controls
        egui::CollapsingHeader::new("LIS Controls")
            .default_open(true)
            .show(ui, |ui| {
                let evt = self.lv_panels.lis_controls.show(ui, state);
                if matches!(evt, LisEvent::RebuildBuffer) {
                    needs_rebuild = true;
                }
            });

        ui.separator();

        // Shape overrides
        egui::CollapsingHeader::new("Shape / Color Overrides")
            .default_open(false)
            .show(ui, |ui| {
                self.lv_panels.shape_overrides.show(ui, state);
            });

        ui.separator();

        // Cluster / ego filter
        egui::CollapsingHeader::new("Cluster Filter")
            .default_open(false)
            .show(ui, |ui| {
                self.lv_panels.cluster_filter.show(ui, state);
            });

        // Export panel
        ui.separator();
        egui::CollapsingHeader::new("Export")
            .default_open(false)
            .show(ui, |ui| {
                self.lv_panels.export_panel.show(ui, state);
            });

        // Audio panel (cfg-gated internally)
        ui.separator();
        egui::CollapsingHeader::new("Audio / MIDI")
            .default_open(false)
            .show(ui, |ui| {
                self.lv_panels.audio_panel.show(ui, state);
            });

        // Surface any error message from state
        if let Some(ref err) = state.load_error.clone() {
            ui.separator();
            ui.colored_label(egui::Color32::RED, format!("Error: {err}"));
        }

        needs_rebuild
    }
}
