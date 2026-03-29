//! `ffmpeg_pipe` — export a frame sequence as video by piping RGBA to ffmpeg.
//!
//! Only available when the `video-export` feature is enabled.

#![cfg(feature = "video-export")]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{bail, Context as _, Result};
use lv_data::{LisBuffer, LisConfig, LvDataset};
use lv_renderer::{ArcballCamera, WgpuContext};

use crate::sequence::export_frame;
use crate::snapshot::SnapshotRenderer;

// ── types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub output_path: PathBuf,
    pub fps: u32,
    pub crf: u32,
    pub codec: String,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("output.mp4"),
            fps: 30,
            crf: 23,
            codec: "libx264".to_string(),
        }
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Export frames `start_frame ..= end_frame` as a video file via ffmpeg.
///
/// Requires `ffmpeg` to be on `PATH`.  RGBA frames are streamed to ffmpeg's
/// stdin; no intermediate files are written.
#[allow(clippy::too_many_arguments)]
pub fn export_video(
    ctx: &WgpuContext,
    dataset: &LvDataset,
    lis_config: &LisConfig,
    buffer: &LisBuffer,
    camera: &ArcballCamera,
    width: u32,
    height: u32,
    start_frame: u32,
    end_frame: u32,
    vid_config: &VideoConfig,
    progress: &mpsc::Sender<f32>,
) -> Result<()> {
    export_video_with_control(
        ctx,
        dataset,
        lis_config,
        buffer,
        camera,
        width,
        height,
        start_frame,
        end_frame,
        vid_config,
        progress,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn export_video_with_control(
    ctx: &WgpuContext,
    dataset: &LvDataset,
    lis_config: &LisConfig,
    buffer: &LisBuffer,
    camera: &ArcballCamera,
    width: u32,
    height: u32,
    start_frame: u32,
    end_frame: u32,
    vid_config: &VideoConfig,
    progress: &mpsc::Sender<f32>,
    cancel: Option<&AtomicBool>,
) -> Result<()> {
    // Validate codec against known-safe values
    const ALLOWED_CODECS: &[&str] = &[
        "libx264",
        "libx265",
        "libvpx-vp9",
        "prores_ks",
        "libaom-av1",
        "h264_nvenc",
        "hevc_nvenc",
    ];
    if !ALLOWED_CODECS.contains(&vid_config.codec.as_str()) {
        bail!(
            "unsupported codec '{}'; allowed: {}",
            vid_config.codec,
            ALLOWED_CODECS.join(", ")
        );
    }
    if vid_config.crf > 63 {
        bail!("crf must be in [0, 63], got {}", vid_config.crf);
    }

    // Verify ffmpeg is available
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("ffmpeg not found on PATH — please install ffmpeg")?;

    let out = vid_config
        .output_path
        .to_str()
        .context("output_path is not valid UTF-8")?;

    // Spawn ffmpeg: read rawvideo RGBA from stdin
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y", // overwrite output
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            &format!("{width}x{height}"),
            "-framerate",
            &vid_config.fps.to_string(),
            "-i",
            "pipe:0", // read from stdin
            "-c:v",
            &vid_config.codec,
            "-crf",
            &vid_config.crf.to_string(),
            "-pix_fmt",
            "yuv420p",
            out,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("spawn ffmpeg")?;

    let stdin = ffmpeg.stdin.as_mut().context("ffmpeg stdin")?;
    if buffer.total_frames == 0 {
        bail!("export_video requires at least one frame");
    }

    let end = end_frame.min(buffer.total_frames.saturating_sub(1));
    if start_frame > end {
        bail!(
            "export_video start_frame {} is out of range for {} total frames",
            start_frame,
            buffer.total_frames
        );
    }
    let total = (end + 1).saturating_sub(start_frame).max(1) as f32;
    let renderer = SnapshotRenderer::new(ctx);

    for frame_idx in start_frame..=end {
        if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            bail!("export_video cancelled");
        }
        let frame = export_frame(dataset, lis_config, buffer, frame_idx)?;

        let img = renderer
            .render_frame(ctx, &frame, camera, width, height)
            .with_context(|| format!("capture_frame {frame_idx}"))?;

        // Write raw RGBA bytes directly to ffmpeg stdin
        stdin
            .write_all(img.as_raw())
            .context("write to ffmpeg stdin")?;

        let pct = (frame_idx + 1 - start_frame) as f32 / total;
        let _ = progress.send(pct);
    }

    // Close stdin to signal EOF
    drop(ffmpeg.stdin.take());

    let status = ffmpeg.wait().context("ffmpeg wait")?;
    if !status.success() {
        let stderr = ffmpeg
            .stderr
            .as_mut()
            .map(|s| {
                use std::io::Read;
                let mut buf = String::new();
                let _ = s.read_to_string(&mut buf);
                buf
            })
            .unwrap_or_default();
        bail!("ffmpeg exited with {status}: {stderr}");
    }

    Ok(())
}
