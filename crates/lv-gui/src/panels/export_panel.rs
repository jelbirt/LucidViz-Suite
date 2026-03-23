//! Export panel — single-frame PNG, image sequence, and (optional) video export.
//!
//! The actual heavy work lives in `lv-export`; this panel provides the UI and
//! fires the work onto a background thread.  When the `export` feature is not
//! compiled in, the panel shows a "not available" notice.

#[cfg(feature = "export")]
mod inner {
    use std::sync::atomic::Ordering;

    use crate::state::{AppState, ExportImageFormat, ExportKind, ExportRequest};

    #[derive(Debug, Default)]
    pub struct ExportPanel;

    impl ExportPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, state: &mut AppState) {
            ui.label("Desktop export is wired for current-frame, image-sequence, and optional video export.");
            ui.separator();

            ui.label("Output directory:");
            ui.horizontal(|ui| {
                let mut output_dir = state
                    .export_output_dir
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_default();
                if ui.text_edit_singleline(&mut output_dir).changed() {
                    state.export_output_dir = if output_dir.trim().is_empty() {
                        None
                    } else {
                        Some(output_dir.trim().into())
                    };
                }
                if ui.button("Browse...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        state.export_output_dir = Some(path);
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Prefix:");
                ui.text_edit_singleline(&mut state.export_filename_prefix);
            });

            ui.horizontal(|ui| {
                ui.label("Width:");
                ui.add(egui::DragValue::new(&mut state.export_width).range(64..=7680));
                ui.label("Height:");
                ui.add(egui::DragValue::new(&mut state.export_height).range(64..=4320));
            });

            egui::ComboBox::from_label("Image format")
                .selected_text(match state.export_format {
                    ExportImageFormat::Png => "PNG",
                    ExportImageFormat::Tga => "TGA",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut state.export_format, ExportImageFormat::Png, "PNG");
                    ui.selectable_value(&mut state.export_format, ExportImageFormat::Tga, "TGA");
                });

            if let Some(buffer) = state.lis_buffer() {
                let max_frame = buffer.total_frames.saturating_sub(1);
                state.export_start_frame = state.export_start_frame.min(max_frame);
                state.export_end_frame = state
                    .export_end_frame
                    .min(max_frame.max(state.export_start_frame));
                ui.horizontal(|ui| {
                    ui.label("Start frame:");
                    ui.add(
                        egui::DragValue::new(&mut state.export_start_frame).range(0..=max_frame),
                    );
                    ui.label("End frame:");
                    ui.add(egui::DragValue::new(&mut state.export_end_frame).range(0..=max_frame));
                });
            }

            ui.horizontal(|ui| {
                ui.label("FPS:");
                ui.add(egui::DragValue::new(&mut state.export_fps).range(1..=240));
                ui.label("CRF:");
                ui.add(egui::DragValue::new(&mut state.export_crf).range(0..=51));
            });
            ui.horizontal(|ui| {
                ui.label("Codec:");
                ui.text_edit_singleline(&mut state.export_codec);
            });

            let can_export = state.lis_buffer().is_some();
            let queue_export = |state: &mut AppState, kind: ExportKind| {
                let Some(output_dir) = state.export_output_dir.clone() else {
                    state.export_status =
                        Some("Choose an output directory before exporting.".into());
                    return;
                };
                state.pending_export_request = Some(ExportRequest {
                    kind,
                    output_dir,
                    filename_prefix: state.export_filename_prefix.trim().to_string(),
                    start_frame: state.export_start_frame,
                    end_frame: state.export_end_frame,
                    width: state.export_width.max(64),
                    height: state.export_height.max(64),
                    format: state.export_format,
                    fps: state.export_fps.max(1),
                    crf: state.export_crf.min(51),
                    codec: state.export_codec.trim().to_string(),
                });
            };

            ui.add_enabled_ui(can_export, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Export current frame").clicked() {
                        queue_export(state, ExportKind::CurrentFrame);
                    }
                    if ui.button("Export image sequence").clicked() {
                        queue_export(state, ExportKind::Sequence);
                    }
                    #[cfg(feature = "video-export")]
                    if ui.button("Export video").clicked() {
                        queue_export(state, ExportKind::Video);
                    }
                });
            });

            #[cfg(not(feature = "video-export"))]
            ui.small(
                "Rebuild with the `video-export` feature to enable in-app ffmpeg video export.",
            );

            if let Some(job) = state.export_job.as_ref() {
                ui.separator();
                ui.label(format!(
                    "Running {:?} export: {:.0}%",
                    job.kind,
                    job.progress * 100.0
                ));
                ui.add(egui::ProgressBar::new(job.progress).show_percentage());
                if ui.button("Cancel export").clicked() {
                    job.cancel_flag.store(true, Ordering::Relaxed);
                    state.export_status = Some("Cancelling export...".into());
                }
            }

            if let Some(ref msg) = state.export_status {
                ui.separator();
                ui.label(msg);
            }
        }
    }
}

#[cfg(not(feature = "export"))]
mod stub {
    use crate::state::AppState;

    #[derive(Default)]
    pub struct ExportPanel;

    impl ExportPanel {
        pub fn show(&mut self, ui: &mut egui::Ui, _state: &mut AppState) {
            ui.label("Export support not compiled in (enable the 'export' feature).");
        }
    }
}

#[cfg(feature = "export")]
pub use inner::ExportPanel;
#[cfg(not(feature = "export"))]
pub use stub::ExportPanel;
