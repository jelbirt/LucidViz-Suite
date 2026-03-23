//! Audio panel — MIDI port selector and playback controls.
//! Only compiled when the `audio` feature is active.

#[cfg(feature = "audio")]
pub use inner::AudioPanel;

#[cfg(not(feature = "audio"))]
pub use stub::AudioPanel;

// ─── Full implementation (feature = "audio") ─────────────────────────────────

#[cfg(feature = "audio")]
mod inner {
    use crate::state::{AppState, AudioRequest};

    #[derive(Default)]
    pub struct AudioPanel;

    impl AudioPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
            ui.heading("MIDI / Audio");
            ui.separator();
            ui.label("Desktop MIDI playback is wired for live LIS-driven sonification, port connect/disconnect, and test tone playback.");
            ui.separator();

            if ui.button("Refresh ports").clicked() {
                state.pending_audio_request = Some(AudioRequest::RefreshPorts);
            }

            if state.audio_selected_port.is_empty() {
                if let Some(first) = state.audio_ports.first() {
                    state.audio_selected_port = first.clone();
                }
            }

            if state.audio_ports.is_empty() {
                ui.label("No MIDI ports found.");
            } else {
                egui::ComboBox::from_label("Port")
                    .selected_text(if state.audio_selected_port.is_empty() {
                        "Select a port"
                    } else {
                        &state.audio_selected_port
                    })
                    .show_ui(ui, |ui| {
                        for port in &state.audio_ports {
                            ui.selectable_value(&mut state.audio_selected_port, port.clone(), port);
                        }
                    });
            }

            let connect_label = if state.audio_connected {
                "Disconnect"
            } else {
                "Connect"
            };
            if ui.button(connect_label).clicked() {
                state.pending_audio_request = Some(if state.audio_connected {
                    AudioRequest::Disconnect
                } else {
                    AudioRequest::Connect(state.audio_selected_port.clone())
                });
            }

            ui.separator();
            ui.checkbox(
                &mut state.audio_live_enabled,
                "Enable live LIS sonification",
            );
            ui.horizontal(|ui| {
                ui.label("Volume:");
                ui.add(egui::Slider::new(&mut state.audio_volume, 0.0..=2.0).fixed_decimals(2));
            });

            ui.horizontal(|ui| {
                ui.label("Beats per transition:");
                ui.add(egui::Slider::new(&mut state.audio_beats, 1..=32));
            });
            ui.horizontal(|ui| {
                ui.label("Hold slices:");
                ui.add(egui::Slider::new(&mut state.audio_hold_slices, 1..=64));
            });

            ui.checkbox(&mut state.audio_graduated, "Graduated Beats");

            if state.audio_graduated {
                ui.horizontal(|ui| {
                    ui.label("Semitone range:");
                    ui.add(egui::Slider::new(&mut state.audio_semitone_range, 1..=24));
                });
            }

            let can_test = state.audio_connected || !state.audio_ports.is_empty();
            ui.add_enabled_ui(can_test, |ui| {
                if ui.button("Test tone (middle C)").clicked() {
                    state.pending_audio_request = Some(AudioRequest::TestTone);
                }
            });

            if let Some(status) = &state.audio_status {
                ui.separator();
                ui.label(status);
            }
        }
    }
}

// ─── No-op stub (no audio feature) ───────────────────────────────────────────

#[cfg(not(feature = "audio"))]
mod stub {
    use crate::state::AppState;

    #[derive(Default)]
    pub struct AudioPanel;

    impl AudioPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, _state: &mut AppState) {
            ui.label("Audio support not compiled in (enable the `audio` feature).");
        }
    }
}
