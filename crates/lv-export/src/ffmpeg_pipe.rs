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

    // ── Validate inputs before spawning ffmpeg ─────────────────────────
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

    // Spawn ffmpeg: read rawvideo RGBA from stdin (all validation passed)
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

/// Check whether ffmpeg is available on PATH.
pub fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Validate a `VideoConfig` without spawning ffmpeg or requiring GPU context.
pub fn validate_video_config(config: &VideoConfig) -> Result<()> {
    const ALLOWED_CODECS: &[&str] = &[
        "libx264",
        "libx265",
        "libvpx-vp9",
        "prores_ks",
        "libaom-av1",
        "h264_nvenc",
        "hevc_nvenc",
    ];
    if !ALLOWED_CODECS.contains(&config.codec.as_str()) {
        bail!(
            "unsupported codec '{}'; allowed: {}",
            config.codec,
            ALLOWED_CODECS.join(", ")
        );
    }
    if config.crf > 63 {
        bail!("crf must be in [0, 63], got {}", config.crf);
    }
    if config.fps == 0 {
        bail!("fps must be > 0");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_config_accepts_valid() {
        let config = VideoConfig::default();
        validate_video_config(&config).expect("default config should be valid");
    }

    #[test]
    fn validate_config_rejects_bad_codec() {
        let config = VideoConfig {
            codec: "malicious; rm -rf /".into(),
            ..Default::default()
        };
        let err = validate_video_config(&config).expect_err("bad codec should fail");
        assert!(err.to_string().contains("unsupported codec"));
    }

    #[test]
    fn validate_config_rejects_high_crf() {
        let config = VideoConfig {
            crf: 100,
            ..Default::default()
        };
        let err = validate_video_config(&config).expect_err("crf > 63 should fail");
        assert!(err.to_string().contains("crf"));
    }

    #[test]
    fn validate_config_rejects_zero_fps() {
        let config = VideoConfig {
            fps: 0,
            ..Default::default()
        };
        let err = validate_video_config(&config).expect_err("fps=0 should fail");
        assert!(err.to_string().contains("fps"));
    }

    #[test]
    fn validate_config_all_allowed_codecs() {
        for codec in [
            "libx264",
            "libx265",
            "libvpx-vp9",
            "prores_ks",
            "libaom-av1",
            "h264_nvenc",
            "hevc_nvenc",
        ] {
            let config = VideoConfig {
                codec: codec.to_string(),
                ..Default::default()
            };
            validate_video_config(&config)
                .unwrap_or_else(|_| panic!("codec '{codec}' should be valid"));
        }
    }

    #[test]
    fn ffmpeg_raw_pipe_protocol() {
        // Skip if ffmpeg is not installed.
        if !ffmpeg_available() {
            eprintln!("SKIPPING ffmpeg_raw_pipe_protocol: ffmpeg not found on PATH");
            return;
        }

        // Pipe 5 frames of 4x4 RGBA data to ffmpeg's null muxer.
        let width = 4u32;
        let height = 4u32;
        let fps = 30;
        let frame_bytes = (width * height * 4) as usize;
        let n_frames = 5;

        let mut ffmpeg = Command::new("ffmpeg")
            .args([
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgba",
                "-video_size",
                &format!("{width}x{height}"),
                "-framerate",
                &fps.to_string(),
                "-i",
                "pipe:0",
                "-f",
                "null",
                "-",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn ffmpeg");

        let stdin = ffmpeg.stdin.as_mut().expect("ffmpeg stdin");

        // Write synthetic frames (solid colors).
        for i in 0..n_frames {
            let pixel = [(i * 50) as u8, 100, 200, 255];
            let frame: Vec<u8> = pixel.iter().copied().cycle().take(frame_bytes).collect();
            stdin.write_all(&frame).expect("write frame to ffmpeg");
        }

        drop(ffmpeg.stdin.take());
        let status = ffmpeg.wait().expect("ffmpeg wait");
        assert!(
            status.success(),
            "ffmpeg should exit successfully with null muxer"
        );
    }
}
