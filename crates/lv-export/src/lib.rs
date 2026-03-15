//! `lv-export` — single-frame PNG snapshot and multi-frame sequence export.
//!
//! Optional `video-export` feature pipes RGBA frames to ffmpeg.

pub mod ffmpeg_pipe;
pub mod sequence;
pub mod snapshot;

#[cfg(feature = "video-export")]
pub use ffmpeg_pipe::{export_video, VideoConfig};
pub use sequence::{capture_sequence, ImageFormat, SequenceConfig};
pub use snapshot::capture_frame;

#[cfg(test)]
mod tests {
    #[test]
    fn smoke() {
        assert_eq!(1 + 1, 2);
    }
}
