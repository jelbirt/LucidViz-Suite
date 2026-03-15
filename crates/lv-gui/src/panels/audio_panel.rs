//! Audio panel — MIDI port selector and playback controls.
//! Only compiled when the `audio` feature is active.

#[cfg(feature = "audio")]
pub use inner::AudioPanel;

#[cfg(not(feature = "audio"))]
pub use stub::AudioPanel;

// ─── Full implementation (feature = "audio") ─────────────────────────────────

#[cfg(feature = "audio")]
mod inner {
    use crate::state::AppState;

    pub struct AudioPanel {
        ports: Vec<String>,
        selected_port: String,
        volume: f32,
        graduated: bool,
        semitone_range: i32,
        connected: bool,
    }

    impl Default for AudioPanel {
        fn default() -> Self {
            Self {
                ports: vec![],
                selected_port: String::new(),
                volume: 1.0,
                graduated: false,
                semitone_range: 12,
                connected: false,
            }
        }
    }

    impl AudioPanel {
        pub fn refresh_ports(&mut self) {
            #[cfg(feature = "audio")]
            {
                self.ports = lv_audio::MidiEngine::list_ports();
                if self.selected_port.is_empty() {
                    if let Some(first) = self.ports.first() {
                        self.selected_port = first.clone();
                    }
                }
            }
        }

        pub fn show(&mut self, ui: &mut egui::Ui, _state: &mut AppState) {
            ui.heading("MIDI / Audio");
            ui.separator();

            if ui.button("Refresh ports").clicked() {
                self.refresh_ports();
            }

            if self.ports.is_empty() {
                ui.label("No MIDI ports found.");
            } else {
                egui::ComboBox::from_label("Port")
                    .selected_text(&self.selected_port)
                    .show_ui(ui, |ui| {
                        for port in &self.ports {
                            ui.selectable_value(&mut self.selected_port, port.clone(), port);
                        }
                    });
            }

            let connect_label = if self.connected {
                "Disconnect"
            } else {
                "Connect"
            };
            if ui.button(connect_label).clicked() {
                self.connected = !self.connected;
            }

            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Volume:");
                ui.add(egui::Slider::new(&mut self.volume, 0.0..=2.0).fixed_decimals(2));
            });

            ui.checkbox(&mut self.graduated, "Graduated Beats");

            if self.graduated {
                ui.horizontal(|ui| {
                    ui.label("Semitone range:");
                    ui.add(egui::Slider::new(&mut self.semitone_range, 1..=24));
                });
            }

            if ui.button("Test tone (middle C)").clicked() {
                // TODO: fire note 60 on channel 0 when engine is wired in Phase 7
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
