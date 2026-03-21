//! LIS (Lucid Interpolation System) buffer builder.
//!
//! Converts an `EtvDataset` into a `LisBuffer` of pre-computed `LisFrame`s, or
//! marks the buffer as streaming if the estimated size exceeds 100 MB.

use lv_data::{EtvDataset, EtvRow, GpuInstance, LisBuffer, LisConfig, LisFrame};

const STREAM_THRESHOLD_BYTES: usize = 100 * 1024 * 1024; // 100 MB

/// Build a `LisBuffer` from `dataset`.
///
/// If the estimated byte count exceeds `STREAM_THRESHOLD_BYTES` the buffer will
/// have `streaming = true` and an empty `frames` vec; callers should use
/// `compute_frame` instead.
pub fn build_lis_buffer(dataset: &EtvDataset, config: &LisConfig) -> LisBuffer {
    let lis = config.lis_value;
    let estimated = dataset.estimated_lis_buffer_bytes(lis);

    if estimated > STREAM_THRESHOLD_BYTES {
        let time_pts = dataset.time_points();
        let total_frames = if time_pts > 1 {
            (time_pts - 1) as u32 * lis
        } else {
            lis
        };
        return LisBuffer {
            frames: vec![],
            streaming: true,
            lis,
            total_frames,
        };
    }

    let time_pts = dataset.time_points();

    if time_pts == 0 {
        return LisBuffer {
            frames: vec![],
            streaming: false,
            lis,
            total_frames: 0,
        };
    }

    let mut frames: Vec<LisFrame> = Vec::new();
    let transitions = if time_pts > 1 { time_pts - 1 } else { 1 };

    for t in 0..transitions {
        let sheet_a = &dataset.sheets[t];
        let sheet_b = if t + 1 < dataset.sheets.len() {
            &dataset.sheets[t + 1]
        } else {
            sheet_a
        };

        for k in 0..lis {
            let alpha = k as f64 / lis as f64;
            let mut labeled_instances: Vec<(String, GpuInstance)> = Vec::new();

            for label in &dataset.all_labels {
                let row_a = sheet_a.rows.iter().find(|r| &r.label == label);
                let row_b = sheet_b.rows.iter().find(|r| &r.label == label);

                if let Some(inst) = interpolate(row_a, row_b, alpha, label) {
                    labeled_instances.push((label.clone(), inst));
                }
            }

            // Sort by shape_id for minimal draw-call switching
            labeled_instances.sort_by_key(|(_, inst)| inst.shape_id);
            let (labels, instances): (Vec<String>, Vec<GpuInstance>) =
                labeled_instances.into_iter().unzip();

            frames.push(LisFrame {
                instances,
                labels,
                slice_index: t as u32 * lis + k,
                transition_index: t as u32,
                local_slice: k,
            });
        }
    }

    let total_frames = frames.len() as u32;
    LisBuffer {
        frames,
        streaming: false,
        lis,
        total_frames,
    }
}

/// Compute a single `LisFrame` on-demand (used in streaming mode).
pub fn compute_frame(dataset: &EtvDataset, config: &LisConfig, slice_index: u32) -> LisFrame {
    let lis = config.lis_value;
    let time_pts = dataset.time_points();
    let transitions = if time_pts > 1 {
        (time_pts - 1) as u32
    } else {
        1
    };

    let transition_index = (slice_index / lis).min(transitions.saturating_sub(1));
    let local_slice = slice_index % lis;
    let alpha = local_slice as f64 / lis as f64;

    let t = transition_index as usize;
    let sheet_a = &dataset.sheets[t];
    let sheet_b = if t + 1 < dataset.sheets.len() {
        &dataset.sheets[t + 1]
    } else {
        sheet_a
    };

    let mut labeled_instances: Vec<(String, GpuInstance)> = Vec::new();
    for label in &dataset.all_labels {
        let row_a = sheet_a.rows.iter().find(|r| &r.label == label);
        let row_b = sheet_b.rows.iter().find(|r| &r.label == label);
        if let Some(inst) = interpolate(row_a, row_b, alpha, label) {
            labeled_instances.push((label.clone(), inst));
        }
    }
    labeled_instances.sort_by_key(|(_, inst)| inst.shape_id);
    let (labels, instances): (Vec<String>, Vec<GpuInstance>) =
        labeled_instances.into_iter().unzip();

    LisFrame {
        instances,
        labels,
        slice_index,
        transition_index,
        local_slice,
    }
}

// ─── Interpolation ────────────────────────────────────────────────────────────

/// Linearly interpolate between `row_a` and `row_b` at `alpha` ∈ [0,1].
///
/// Missing rows are treated as fade-in (size=0 at the missing end).
fn interpolate(
    row_a: Option<&EtvRow>,
    row_b: Option<&EtvRow>,
    alpha: f64,
    _label: &str,
) -> Option<GpuInstance> {
    match (row_a, row_b) {
        (None, None) => None,
        (Some(a), None) => {
            // Fade out: shrink to zero
            let size = lerp(a.size, 0.0, alpha) as f32;
            let pos = [a.x as f32, a.y as f32, a.z as f32];
            Some(GpuInstance::from_row(a, pos).with_size(size))
        }
        (None, Some(b)) => {
            // Fade in: grow from zero
            let size = lerp(0.0, b.size, alpha) as f32;
            let pos = [b.x as f32, b.y as f32, b.z as f32];
            Some(GpuInstance::from_row(b, pos).with_size(size))
        }
        (Some(a), Some(b)) => {
            let x = lerp(a.x, b.x, alpha) as f32;
            let y = lerp(a.y, b.y, alpha) as f32;
            let z = lerp(a.z, b.z, alpha) as f32;
            let size = lerp(a.size, b.size, alpha) as f32;
            let sa = lerp(a.size_alpha, b.size_alpha, alpha) as f32;
            let sx = lerp(a.spin_x, b.spin_x, alpha) as f32;
            let sy = lerp(a.spin_y, b.spin_y, alpha) as f32;
            let sz = lerp(a.spin_z, b.spin_z, alpha) as f32;
            let r = lerp(a.color_r as f64, b.color_r as f64, alpha) as f32;
            let g = lerp(a.color_g as f64, b.color_g as f64, alpha) as f32;
            let bv = lerp(a.color_b as f64, b.color_b as f64, alpha) as f32;

            // Use shape from `a` (transitions don't morph shapes)
            let mut inst = GpuInstance::from_row(a, [x, y, z]);
            inst.size = size;
            inst.size_alpha = sa;
            inst.spin = [sx, sy, sz];
            inst.color = [r, g, bv, 1.0];
            Some(inst)
        }
    }
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ─── GpuInstance helpers (extension trait for size override) ─────────────────

trait GpuInstanceExt {
    fn with_size(self, size: f32) -> GpuInstance;
}

impl GpuInstanceExt for GpuInstance {
    fn with_size(mut self, size: f32) -> GpuInstance {
        self.size = size;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lv_data::{EtvDataset, EtvRow, EtvSheet, ShapeKind};

    fn make_dataset() -> EtvDataset {
        let sheet = EtvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows: vec![
                EtvRow {
                    label: "cube".into(),
                    shape: ShapeKind::Cube,
                    x: 4.0,
                    ..Default::default()
                },
                EtvRow {
                    label: "sphere".into(),
                    shape: ShapeKind::Sphere,
                    x: 1.0,
                    ..Default::default()
                },
            ],
            edges: vec![],
        };

        EtvDataset {
            source_path: None,
            sheets: vec![sheet],
            all_labels: vec!["cube".into(), "sphere".into()],
        }
    }

    #[test]
    fn build_lis_buffer_keeps_labels_aligned_with_sorted_instances() {
        let buffer = build_lis_buffer(
            &make_dataset(),
            &LisConfig {
                lis_value: 4,
                ..Default::default()
            },
        );

        let frame = &buffer.frames[0];
        assert_eq!(frame.labels, vec!["sphere", "cube"]);
        assert_eq!(frame.instances.len(), frame.labels.len());
        assert_eq!(frame.instances[0].shape_id, ShapeKind::Sphere.gpu_id());
        assert_eq!(frame.instances[1].shape_id, ShapeKind::Cube.gpu_id());
        assert_eq!(frame.instances[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(frame.instances[1].position, [4.0, 0.0, 0.0]);
    }

    #[test]
    fn compute_frame_keeps_labels_aligned_with_instances() {
        let frame = compute_frame(
            &make_dataset(),
            &LisConfig {
                lis_value: 4,
                ..Default::default()
            },
            0,
        );

        assert_eq!(frame.labels, vec!["sphere", "cube"]);
        assert_eq!(frame.instances.len(), frame.labels.len());
    }
}
