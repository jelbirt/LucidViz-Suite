//! `sequence` — export a range of LIS frames as individual image files.

use std::path::PathBuf;
use std::sync::mpsc;

use anyhow::{Context as _, Result};
use lv_data::LisBuffer;
use lv_renderer::{ArcballCamera, WgpuContext};

use crate::snapshot::capture_frame;

// ── types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Png,
    Tga,
}

impl ImageFormat {
    fn extension(self) -> &'static str {
        match self {
            ImageFormat::Png => "png",
            ImageFormat::Tga => "tga",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SequenceConfig {
    pub output_dir: PathBuf,
    pub filename_prefix: String,
    pub start_frame: u32,
    pub end_frame: u32,
    pub width: u32,
    pub height: u32,
    pub format: ImageFormat,
}

// ── public API ────────────────────────────────────────────────────────────────

/// Export `config.start_frame ..= config.end_frame` as numbered image files.
///
/// Progress is reported as a `f32` in `[0.0, 1.0]` on `progress`.
/// Errors are returned immediately; partial output may exist on disk.
pub fn capture_sequence(
    ctx: &WgpuContext,
    buffer: &LisBuffer,
    camera: &ArcballCamera,
    config: &SequenceConfig,
    progress: &mpsc::Sender<f32>,
) -> Result<()> {
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create output dir {:?}", config.output_dir))?;

    let start = config.start_frame;
    let end = config
        .end_frame
        .min(buffer.frames.len().saturating_sub(1) as u32);
    let total = (end + 1).saturating_sub(start).max(1) as f32;
    let ext = config.format.extension();

    for frame_idx in start..=end {
        let frame = buffer
            .frames
            .get(frame_idx as usize)
            .with_context(|| format!("frame {frame_idx} out of range"))?;

        let img = capture_frame(ctx, frame, camera, config.width, config.height)
            .with_context(|| format!("capture_frame {frame_idx}"))?;

        let filename = format!("{}_{frame_idx:06}.{ext}", config.filename_prefix);
        let path = config.output_dir.join(&filename);

        img.save(&path).with_context(|| format!("save {path:?}"))?;

        let pct = (frame_idx + 1 - start) as f32 / total;
        let _ = progress.send(pct);
    }

    Ok(())
}
