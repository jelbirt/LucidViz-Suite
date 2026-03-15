//! LIS transport and animation controls panel.

use crate::state::{AppState, PlayState};

/// Events emitted by the LIS control panel.
pub enum LisEvent {
    /// The LIS value changed — caller must rebuild the LisBuffer.
    RebuildBuffer,
    None,
}

/// Animation transport and LIS configuration panel.
#[derive(Default)]
pub struct LisControlPanel;

impl LisControlPanel {
    pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) -> LisEvent {
        let mut ev = LisEvent::None;

        ui.heading("LIS Controls");
        ui.separator();

        // ── LIS value ──────────────────────────────────────────────────────
        let prev_lis = state.lis_config.lis_value;
        ui.horizontal(|ui| {
            ui.label("LIS frames:");
            ui.add(egui::Slider::new(&mut state.lis_config.lis_value, 2..=300).integer());
        });
        if state.lis_config.lis_value != prev_lis {
            ev = LisEvent::RebuildBuffer;
        }

        // ── Speed ──────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Speed:");
            ui.add(
                egui::Slider::new(&mut state.lis_config.speed, 0.1..=10.0)
                    .logarithmic(true)
                    .fixed_decimals(2),
            );
        });

        // ── FPS ────────────────────────────────────────────────────────────
        let fps_options = ["Vsync", "24", "30", "60", "120"];
        let current_fps_label = match state.lis_config.target_fps {
            None => "Vsync",
            Some(24) => "24",
            Some(30) => "30",
            Some(60) => "60",
            Some(120) => "120",
            Some(_) => "Custom",
        };
        egui::ComboBox::from_label("FPS")
            .selected_text(current_fps_label)
            .show_ui(ui, |ui| {
                for opt in fps_options {
                    let fps_val = match opt {
                        "Vsync" => None,
                        s => s.parse::<u32>().ok(),
                    };
                    ui.selectable_value(&mut state.lis_config.target_fps, fps_val, opt);
                }
                ui.horizontal(|ui| {
                    ui.label("Custom:");
                    let mut custom = state.lis_config.target_fps.unwrap_or(60);
                    if ui
                        .add(egui::DragValue::new(&mut custom).range(1..=999))
                        .changed()
                    {
                        state.lis_config.target_fps = Some(custom);
                    }
                });
            });

        // ── Transport ──────────────────────────────────────────────────────
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("▶").clicked() {
                state.play_state = PlayState::Playing;
            }
            if ui.button("⏸").clicked() {
                state.play_state = PlayState::Paused;
            }
            if ui.button("⏹").clicked() {
                state.play_state = PlayState::Stopped;
                state.slice_index = 0;
            }
            ui.checkbox(&mut state.lis_config.looping, "Loop");
        });

        // ── Scrubber ───────────────────────────────────────────────────────
        if let Some(buf) = &state.lis_buffer {
            let total = buf.total_frames.max(1);
            ui.horizontal(|ui| {
                ui.label(format!("Frame {}/{total}", state.slice_index + 1));
            });
            ui.add(egui::Slider::new(&mut state.slice_index, 0..=(total - 1)));
        }

        ev
    }
}
