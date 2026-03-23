//! `lv-export` — single-frame PNG snapshot and multi-frame sequence export.
//!
//! Optional `video-export` feature pipes RGBA frames to ffmpeg.

pub mod ffmpeg_pipe;
pub mod sequence;
pub mod snapshot;

#[cfg(feature = "video-export")]
pub use ffmpeg_pipe::{export_video, export_video_with_control, VideoConfig};
pub use sequence::{
    capture_sequence, capture_sequence_with_control, ImageFormat, SequenceConfig, SequenceControl,
};
pub use snapshot::capture_frame;

#[cfg(test)]
mod tests {
    use crate::sequence::export_frame;
    use lv_data::{EtvDataset, EtvRow, EtvSheet, LisBuffer, LisConfig, ShapeKind};

    fn dataset() -> EtvDataset {
        EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "T0".into(),
                sheet_index: 0,
                rows: vec![EtvRow {
                    label: "alpha".into(),
                    shape: ShapeKind::Sphere,
                    ..Default::default()
                }],
                edges: vec![],
            }],
            all_labels: vec!["alpha".into()],
        }
    }

    #[test]
    fn smoke() {
        assert_eq!(1 + 1, 2);
    }

    #[test]
    fn export_frame_reports_empty_buffers() {
        let ds = dataset();
        let cfg = LisConfig::default();
        let buffer = LisBuffer {
            frames: vec![],
            streaming: true,
            lis: cfg.lis_value,
            total_frames: 0,
        };

        let err = export_frame(&ds, &cfg, &buffer, 0).expect_err("empty buffers must fail");
        assert!(err.to_string().contains("out of range"));
    }
}
