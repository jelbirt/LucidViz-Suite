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
            crate::help_marker(
                ui,
                "Number of interpolated frames between each pair of time points. \
                 Higher values produce smoother transitions but use more memory.",
            );
            ui.add(egui::Slider::new(&mut state.lis_config.lis_value, 2..=300).integer());
        });
        if state.lis_config.lis_value != prev_lis {
            ev = LisEvent::RebuildBuffer;
        }

        // ── Speed ──────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Speed:");
            crate::help_marker(
                ui,
                "Playback speed multiplier. 1.0 = normal, 2.0 = double speed, \
                 0.5 = half speed. Logarithmic scale.",
            );
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

        // ── Easing ─────────────────────────────────────────────────────────
        let prev_easing = state.lis_config.easing;
        crate::help_marker(
            ui,
            "Easing controls the acceleration curve of interpolation between time slices. \
             Linear = constant speed. Ease In = slow start. Ease Out = slow end. \
             Ease In-Out = slow start and end. Spring = overshoot and settle.",
        );
        egui::ComboBox::from_label("Easing")
            .selected_text(state.lis_config.easing.to_string())
            .show_ui(ui, |ui| {
                for &mode in lv_data::EasingMode::ALL {
                    ui.selectable_value(&mut state.lis_config.easing, mode, mode.to_string());
                }
            });
        if state.lis_config.easing != prev_easing {
            ev = LisEvent::RebuildBuffer;
        }

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
                state.pending_slice_index = Some(0);
            }
            ui.checkbox(&mut state.lis_config.looping, "Loop");
        });

        // ── Scrubber ───────────────────────────────────────────────────────
        if let Some(buf) = state.lis_buffer() {
            let total = buf.total_frames.max(1);
            let mut requested_slice = state.pending_slice_index.unwrap_or(state.slice_index());
            ui.horizontal(|ui| {
                ui.label(format!("Frame {}/{total}", requested_slice + 1));
            });
            if ui
                .add(egui::Slider::new(&mut requested_slice, 0..=(total - 1)))
                .changed()
            {
                state.pending_slice_index = Some(requested_slice);
            }
        }

        ev
    }
}
