//! `sequence` — export a range of LIS frames as individual image files.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use anyhow::{bail, Context as _, Result};
use lv_data::{EtvDataset, LisBuffer, LisConfig, LisFrame};
use lv_renderer::{compute_frame, ArcballCamera, WgpuContext};

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

#[derive(Debug, Clone, Copy)]
pub struct SequenceControl<'a> {
    pub progress: &'a mpsc::Sender<f32>,
    pub cancel: Option<&'a AtomicBool>,
}

// ── public API ────────────────────────────────────────────────────────────────

/// Export `config.start_frame ..= config.end_frame` as numbered image files.
///
/// Progress is reported as a `f32` in `[0.0, 1.0]` on `progress`.
/// Errors are returned immediately; partial output may exist on disk.
pub fn capture_sequence(
    ctx: &WgpuContext,
    dataset: &EtvDataset,
    lis_config: &LisConfig,
    buffer: &LisBuffer,
    camera: &ArcballCamera,
    config: &SequenceConfig,
    progress: &mpsc::Sender<f32>,
) -> Result<()> {
    capture_sequence_with_control(
        ctx,
        dataset,
        lis_config,
        buffer,
        camera,
        config,
        SequenceControl {
            progress,
            cancel: None,
        },
    )
}

pub fn capture_sequence_with_control(
    ctx: &WgpuContext,
    dataset: &EtvDataset,
    lis_config: &LisConfig,
    buffer: &LisBuffer,
    camera: &ArcballCamera,
    config: &SequenceConfig,
    control: SequenceControl<'_>,
) -> Result<()> {
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create output dir {:?}", config.output_dir))?;

    let total_frames = buffer.total_frames;
    if total_frames == 0 {
        bail!("capture_sequence requires at least one frame");
    }

    let start = config.start_frame;
    let end = config.end_frame.min(total_frames.saturating_sub(1));
    if start > end {
        bail!(
            "capture_sequence start_frame {} is out of range for {} total frames",
            start,
            total_frames
        );
    }
    let total = (end + 1).saturating_sub(start).max(1) as f32;
    let ext = config.format.extension();

    for frame_idx in start..=end {
        if control
            .cancel
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            bail!("capture_sequence cancelled");
        }
        let frame = export_frame(dataset, lis_config, buffer, frame_idx)?;

        let img = capture_frame(ctx, &frame, camera, config.width, config.height)
            .with_context(|| format!("capture_frame {frame_idx}"))?;

        let filename = format!("{}_{frame_idx:06}.{ext}", config.filename_prefix);
        let path = config.output_dir.join(&filename);

        img.save(&path).with_context(|| format!("save {path:?}"))?;

        let pct = (frame_idx + 1 - start) as f32 / total;
        let _ = control.progress.send(pct);
    }

    Ok(())
}

pub(crate) fn export_frame(
    dataset: &EtvDataset,
    lis_config: &LisConfig,
    buffer: &LisBuffer,
    frame_idx: u32,
) -> Result<LisFrame> {
    if frame_idx >= buffer.total_frames {
        bail!(
            "frame {frame_idx} out of range for {} total frames",
            buffer.total_frames
        );
    }

    if buffer.streaming || buffer.frames.is_empty() {
        Ok(compute_frame(dataset, lis_config, frame_idx))
    } else {
        buffer
            .frames
            .get(frame_idx as usize)
            .cloned()
            .with_context(|| format!("frame {frame_idx} out of range"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        capture_sequence_with_control, export_frame, ImageFormat, SequenceConfig, SequenceControl,
    };
    use lv_data::{EtvDataset, EtvRow, EtvSheet, LisBuffer, LisConfig, ShapeKind};
    use lv_renderer::{build_lis_buffer, compute_frame};
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    fn sample_dataset() -> EtvDataset {
        EtvDataset {
            source_path: None,
            sheets: vec![
                EtvSheet {
                    name: "T0".into(),
                    sheet_index: 0,
                    rows: vec![EtvRow {
                        label: "alpha".into(),
                        shape: ShapeKind::Sphere,
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        ..Default::default()
                    }],
                    edges: vec![],
                },
                EtvSheet {
                    name: "T1".into(),
                    sheet_index: 1,
                    rows: vec![EtvRow {
                        label: "alpha".into(),
                        shape: ShapeKind::Sphere,
                        x: 10.0,
                        y: 5.0,
                        z: -2.0,
                        ..Default::default()
                    }],
                    edges: vec![],
                },
            ],
            all_labels: vec!["alpha".into()],
        }
    }

    #[test]
    fn export_frame_uses_precomputed_frames_when_present() {
        let dataset = sample_dataset();
        let lis_config = LisConfig {
            lis_value: 4,
            ..Default::default()
        };
        let buffer = build_lis_buffer(&dataset, &lis_config);

        let frame = export_frame(&dataset, &lis_config, &buffer, 2).expect("frame should resolve");

        assert_eq!(frame.slice_index, buffer.frames[2].slice_index);
        assert_eq!(frame.transition_index, buffer.frames[2].transition_index);
        assert_eq!(frame.local_slice, buffer.frames[2].local_slice);
        assert_eq!(frame.labels, buffer.frames[2].labels);
        assert_eq!(frame.instances, buffer.frames[2].instances);
    }

    #[test]
    fn export_frame_computes_streaming_frames_on_demand() {
        let dataset = sample_dataset();
        let lis_config = LisConfig {
            lis_value: 4,
            ..Default::default()
        };
        let expected = compute_frame(&dataset, &lis_config, 1);
        let buffer = LisBuffer {
            frames: vec![],
            streaming: true,
            lis: lis_config.lis_value,
            total_frames: 4,
        };

        let frame = export_frame(&dataset, &lis_config, &buffer, 1).expect("frame should resolve");

        assert_eq!(frame.slice_index, expected.slice_index);
        assert_eq!(frame.transition_index, expected.transition_index);
        assert_eq!(frame.local_slice, expected.local_slice);
        assert_eq!(frame.labels, expected.labels);
        assert_eq!(frame.instances, expected.instances);
    }

    #[test]
    fn export_frame_rejects_out_of_range_indices() {
        let dataset = sample_dataset();
        let lis_config = LisConfig {
            lis_value: 4,
            ..Default::default()
        };
        let buffer = LisBuffer {
            frames: vec![],
            streaming: true,
            lis: lis_config.lis_value,
            total_frames: 4,
        };

        let err = export_frame(&dataset, &lis_config, &buffer, 4)
            .expect_err("out-of-range frame must fail");

        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn capture_sequence_honors_cancellation_before_rendering() {
        let dataset = sample_dataset();
        let lis_config = LisConfig {
            lis_value: 4,
            ..Default::default()
        };
        let buffer = build_lis_buffer(&dataset, &lis_config);
        let progress_tx = mpsc::channel().0;
        let cancel = AtomicBool::new(true);
        let Ok(ctx) = lv_renderer::WgpuContext::new_headless() else {
            return;
        };
        let camera = lv_renderer::ArcballCamera::new(1.0);
        let tempdir = tempfile::tempdir().expect("tempdir");
        let config = SequenceConfig {
            output_dir: tempdir.path().to_path_buf(),
            filename_prefix: "cancel".into(),
            start_frame: 0,
            end_frame: 1,
            width: 128,
            height: 128,
            format: ImageFormat::Png,
        };

        let err = capture_sequence_with_control(
            &ctx,
            &dataset,
            &lis_config,
            &buffer,
            &camera,
            &config,
            SequenceControl {
                progress: &progress_tx,
                cancel: Some(&cancel),
            },
        )
        .expect_err("cancelled export should abort");

        assert!(err.to_string().contains("cancelled"));
    }
}
