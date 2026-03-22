//! Export panel — single-frame PNG, image sequence, and (optional) video export.
//!
//! The actual heavy work lives in `lv-export`; this panel provides the UI and
//! fires the work onto a background thread.  When the `export` feature is not
//! compiled in, the panel shows a "not available" notice.

#[cfg(feature = "export")]
mod inner {
    use std::path::PathBuf;
    use std::sync::mpsc;

    use crate::state::AppState;

    /// Export event returned from [`ExportPanel::show`].
    #[derive(Debug)]
    pub enum ExportEvent {
        /// Sequence or video export started (background thread launched).
        JobStarted,
        /// Single-frame export completed (path written).
        FrameSaved(PathBuf),
        /// An error occurred.
        Error(String),
        None,
    }

    #[derive(Debug, Default)]
    pub struct ExportPanel {
        // Sequence settings
        pub output_dir: String,
        pub filename_prefix: String,
        pub start_frame: u32,
        pub end_frame: u32,
        pub width: u32,
        pub height: u32,
        pub format_png: bool, // true=PNG, false=TGA

        // Video settings (video-export feature)
        pub fps: u32,
        pub crf: u32,
        pub codec: String,

        // Progress feedback
        pub running: bool,
        pub progress: f32,
        pub last_msg: Option<String>,
        pub receiver: Option<mpsc::Receiver<Result<(), String>>>,
        pub prog_rx: Option<mpsc::Receiver<f32>>,
    }

    impl ExportPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, state: &AppState) {
            ui.label("Export integration is not wired into `lv-app` yet.");
            ui.small("The `lv-export` crate and export demo work, but this GUI panel is currently informational only.");
            ui.separator();

            // Poll background results
            if self.running {
                if let Some(rx) = &self.prog_rx {
                    while let Ok(p) = rx.try_recv() {
                        self.progress = p;
                    }
                }
                if let Some(rx) = &self.receiver {
                    if let Ok(result) = rx.try_recv() {
                        self.running = false;
                        self.progress = 1.0;
                        match result {
                            Ok(()) => self.last_msg = Some("Export complete.".into()),
                            Err(e) => self.last_msg = Some(format!("Error: {e}")),
                        }
                    }
                }
            }

            ui.add_enabled_ui(false, |ui| {
                ui.label("Output directory:");
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(&mut self.output_dir);
                    let _ = ui.button("Browse...");
                });

                ui.horizontal(|ui| {
                    ui.label("Prefix:");
                    ui.text_edit_singleline(&mut self.filename_prefix);
                });

                ui.horizontal(|ui| {
                    ui.label("Start frame:");
                    ui.add(egui::DragValue::new(&mut self.start_frame));
                    ui.label("End:");
                    ui.add(egui::DragValue::new(&mut self.end_frame));
                });

                ui.horizontal(|ui| {
                    ui.label("Width:");
                    ui.add(egui::DragValue::new(&mut self.width).range(64..=7680));
                    ui.label("Height:");
                    ui.add(egui::DragValue::new(&mut self.height).range(64..=4320));
                });

                ui.checkbox(&mut self.format_png, "PNG (uncheck for TGA)");

                ui.separator();

                // Single-frame export (sync)
                let can_export = state.lis_buffer.is_some() && !self.running;
                ui.add_enabled_ui(can_export, |ui| {
                    let _ = ui.button("Export current frame (PNG)");
                });

                // Sequence export (background)
                ui.add_enabled_ui(can_export, |ui| {
                    let _ = ui.button("Export sequence...");
                });

                // Video-export controls are only compiled when the video-export
                // sub-feature is active.
                #[cfg(feature = "video-export")]
                {
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("FPS:");
                        ui.add(egui::DragValue::new(&mut self.fps).range(1..=120));
                        ui.label("CRF:");
                        ui.add(egui::DragValue::new(&mut self.crf).range(0..=51));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Codec:");
                        ui.text_edit_singleline(&mut self.codec);
                    });
                    ui.add_enabled_ui(can_export, |ui| {
                        let _ = ui.button("Export video (requires ffmpeg)...");
                    });
                }
            });

            let _ = (&state.lis_buffer, PathBuf::from(&self.output_dir));
            self.last_msg.get_or_insert_with(|| {
                "Not available in the desktop app yet; use `cargo run -p lv-export --bin export_demo` for a working export flow.".into()
            });

            // Progress / status
            if self.running {
                ui.add(egui::ProgressBar::new(self.progress).show_percentage());
            }
            if let Some(ref msg) = self.last_msg {
                ui.label(msg);
            }
        }
    }

    impl ExportPanel {}
}

#[cfg(not(feature = "export"))]
mod stub {
    use crate::state::AppState;

    #[derive(Default)]
    pub struct ExportPanel;

    impl ExportPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, _state: &AppState) {
            ui.label("Export support not compiled in (enable the 'export' feature).");
        }
    }
}

#[cfg(feature = "export")]
pub use inner::ExportPanel;
#[cfg(not(feature = "export"))]
pub use stub::ExportPanel;
