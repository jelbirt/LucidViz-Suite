//! LIS (Lucid Interpolation System) buffer builder.
//!
//! Converts an `LvDataset` into a `LisBuffer` of pre-computed `LisFrame`s, or
//! marks the buffer as streaming if the estimated size exceeds 100 MB.

use std::collections::HashMap;

use rayon::prelude::*;

use lv_data::{GpuInstance, LisBuffer, LisConfig, LisFrame, LvDataset, LvRow, LvSheet};

const STREAM_THRESHOLD_BYTES: usize = 100 * 1024 * 1024; // 100 MB

/// Build a `LisBuffer` from `dataset`.
///
/// If the estimated byte count exceeds `STREAM_THRESHOLD_BYTES` the buffer will
/// have `streaming = true` and an empty `frames` vec; callers should use
/// `compute_frame` instead.
pub fn build_lis_buffer(dataset: &LvDataset, config: &LisConfig) -> LisBuffer {
    let lis = config.lis_value;

    // Guard: empty dataset gets an empty non-streaming buffer regardless of
    // estimated size, so `compute_frame` is never called with no sheets.
    let time_pts = dataset.time_points();
    if time_pts == 0 {
        return LisBuffer {
            frames: vec![],
            streaming: false,
            lis,
            total_frames: 0,
        };
    }

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

    // time_pts > 0 is guaranteed by the early return above.
    let indexed_sheets: Vec<HashMap<&str, &LvRow>> =
        dataset.sheets.iter().map(index_sheet_rows).collect();
    let transitions = if time_pts > 1 { time_pts - 1 } else { 1 };

    // Parallelize across transitions: each transition's frames are independent.
    let easing = config.easing;
    let frames: Vec<LisFrame> = (0..transitions)
        .into_par_iter()
        .flat_map(|t| {
            let rows_a = &indexed_sheets[t];
            let rows_b = if t + 1 < indexed_sheets.len() {
                &indexed_sheets[t + 1]
            } else {
                rows_a
            };

            (0..lis)
                .map(|k| {
                    let linear_alpha = k as f64 / lis as f64;
                    let alpha = easing.apply(linear_alpha);
                    build_frame(
                        dataset,
                        rows_a,
                        rows_b,
                        alpha,
                        t as u32 * lis + k,
                        t as u32,
                        k,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let total_frames = frames.len() as u32;
    LisBuffer {
        frames,
        streaming: false,
        lis,
        total_frames,
    }
}

/// A small LRU cache for streaming mode to avoid recomputing recent frames.
///
/// Uses the `lru` crate for O(1) amortized get/put/eviction instead of
/// the previous Vec-based O(n) approach.
pub struct FrameCache {
    inner: lru::LruCache<u32, LisFrame>,
}

impl FrameCache {
    /// Create a new frame cache with the given capacity (number of frames to keep).
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: lru::LruCache::new(std::num::NonZeroUsize::new(capacity.max(1)).unwrap()),
        }
    }

    /// Get a cached frame, or compute and cache it.
    pub fn get_or_compute(
        &mut self,
        dataset: &LvDataset,
        config: &LisConfig,
        slice_index: u32,
    ) -> &LisFrame {
        // O(1) lookup + promotion to MRU position.
        if self.inner.contains(&slice_index) {
            return self.inner.get(&slice_index).unwrap();
        }

        // Compute, insert (auto-evicts LRU if at capacity), and return ref.
        let frame = compute_frame(dataset, config, slice_index);
        self.inner.put(slice_index, frame);
        self.inner.get(&slice_index).unwrap()
    }

    /// Clear all cached frames.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

/// Compute a single `LisFrame` on-demand (used in streaming mode).
pub fn compute_frame(dataset: &LvDataset, config: &LisConfig, slice_index: u32) -> LisFrame {
    let lis = config.lis_value;
    let time_pts = dataset.time_points();
    let transitions = if time_pts > 1 {
        (time_pts - 1) as u32
    } else {
        1
    };

    let transition_index = (slice_index / lis).min(transitions.saturating_sub(1));
    let local_slice = slice_index % lis;
    let linear_alpha = local_slice as f64 / lis as f64;
    let alpha = config.easing.apply(linear_alpha);

    let t = transition_index as usize;
    let rows_a = index_sheet_rows(&dataset.sheets[t]);
    let rows_b = if t + 1 < dataset.sheets.len() {
        index_sheet_rows(&dataset.sheets[t + 1])
    } else {
        index_sheet_rows(&dataset.sheets[t])
    };

    build_frame(
        dataset,
        &rows_a,
        &rows_b,
        alpha,
        slice_index,
        transition_index,
        local_slice,
    )
}

fn index_sheet_rows(sheet: &LvSheet) -> HashMap<&str, &LvRow> {
    sheet
        .rows
        .iter()
        .map(|row| (row.label.as_str(), row))
        .collect()
}

fn build_frame(
    dataset: &LvDataset,
    rows_a: &HashMap<&str, &LvRow>,
    rows_b: &HashMap<&str, &LvRow>,
    alpha: f64,
    slice_index: u32,
    transition_index: u32,
    local_slice: u32,
) -> LisFrame {
    let mut labeled_instances: Vec<(String, GpuInstance)> = Vec::new();
    for label in &dataset.all_labels {
        let row_a = rows_a.get(label.as_str()).copied();
        let row_b = rows_b.get(label.as_str()).copied();
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
    row_a: Option<&LvRow>,
    row_b: Option<&LvRow>,
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
            let sx = lerp_angle(a.spin_x, b.spin_x, alpha) as f32;
            let sy = lerp_angle(a.spin_y, b.spin_y, alpha) as f32;
            let sz = lerp_angle(a.spin_z, b.spin_z, alpha) as f32;
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

/// Interpolate between two angles (in degrees) via the shortest arc.
fn lerp_angle(a: f64, b: f64, t: f64) -> f64 {
    let mut diff = (b - a) % 360.0;
    if diff > 180.0 {
        diff -= 360.0;
    }
    if diff < -180.0 {
        diff += 360.0;
    }
    a + diff * t
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
    use lv_data::{LvDataset, LvRow, LvSheet, ShapeKind};

    fn make_dataset() -> LvDataset {
        let sheet = LvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows: vec![
                LvRow {
                    label: "cube".into(),
                    shape: ShapeKind::Cube,
                    x: 4.0,
                    ..Default::default()
                },
                LvRow {
                    label: "sphere".into(),
                    shape: ShapeKind::Sphere,
                    x: 1.0,
                    ..Default::default()
                },
            ],
            edges: vec![],
        };

        LvDataset {
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

    fn make_two_sheet_dataset() -> LvDataset {
        let sheet_a = LvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows: vec![LvRow {
                label: "node".into(),
                shape: ShapeKind::Sphere,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                size: 2.0,
                ..Default::default()
            }],
            edges: vec![],
        };
        let sheet_b = LvSheet {
            name: "T1".into(),
            sheet_index: 1,
            rows: vec![LvRow {
                label: "node".into(),
                shape: ShapeKind::Sphere,
                x: 10.0,
                y: 20.0,
                z: 30.0,
                size: 4.0,
                ..Default::default()
            }],
            edges: vec![],
        };
        LvDataset {
            source_path: None,
            sheets: vec![sheet_a, sheet_b],
            all_labels: vec!["node".into()],
        }
    }

    fn lis4() -> LisConfig {
        LisConfig {
            lis_value: 4,
            ..Default::default()
        }
    }

    #[test]
    fn interpolate_at_alpha_zero_returns_source_position() {
        let ds = make_two_sheet_dataset();
        let frame = compute_frame(&ds, &lis4(), 0);
        let inst = &frame.instances[0];
        assert_eq!(inst.position, [0.0, 0.0, 0.0]);
        assert!((inst.size - 2.0).abs() < 1e-6);
    }

    #[test]
    fn interpolate_at_alpha_half_returns_midpoint() {
        let ds = make_two_sheet_dataset();
        // lis_value=4, frame 2 => alpha = 2/4 = 0.5
        let frame = compute_frame(&ds, &lis4(), 2);
        let inst = &frame.instances[0];
        assert!(
            (inst.position[0] - 5.0).abs() < 1e-6,
            "x midpoint: {}",
            inst.position[0]
        );
        assert!(
            (inst.position[1] - 10.0).abs() < 1e-6,
            "y midpoint: {}",
            inst.position[1]
        );
        assert!(
            (inst.position[2] - 15.0).abs() < 1e-6,
            "z midpoint: {}",
            inst.position[2]
        );
        assert!(
            (inst.size - 3.0).abs() < 1e-6,
            "size midpoint: {}",
            inst.size
        );
    }

    #[test]
    fn frame_cache_returns_same_result() {
        let ds = make_two_sheet_dataset();
        let cfg = lis4();
        let mut cache = FrameCache::new(4);

        let frame1 = compute_frame(&ds, &cfg, 2);
        let cached = cache.get_or_compute(&ds, &cfg, 2);
        assert_eq!(cached.slice_index, frame1.slice_index);
        assert_eq!(cached.instances.len(), frame1.instances.len());
        assert_eq!(cached.instances[0].position, frame1.instances[0].position);

        // Second access should hit cache.
        let cached2 = cache.get_or_compute(&ds, &cfg, 2);
        assert_eq!(cached2.slice_index, 2);
    }

    #[test]
    fn frame_cache_evicts_oldest() {
        let ds = make_two_sheet_dataset();
        let cfg = lis4();
        let mut cache = FrameCache::new(2);

        cache.get_or_compute(&ds, &cfg, 0);
        cache.get_or_compute(&ds, &cfg, 1);
        assert_eq!(cache.inner.len(), 2);

        // Adding a third should evict frame 0.
        cache.get_or_compute(&ds, &cfg, 2);
        assert_eq!(cache.inner.len(), 2);
        assert!(!cache.inner.contains(&0));
    }

    #[test]
    fn fade_in_interpolation_starts_at_zero_size() {
        // Node exists only in sheet B, so at alpha=0 it should fade in from size=0
        let sheet_a = LvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows: vec![],
            edges: vec![],
        };
        let sheet_b = LvSheet {
            name: "T1".into(),
            sheet_index: 1,
            rows: vec![LvRow {
                label: "appearing".into(),
                shape: ShapeKind::Cube,
                x: 5.0,
                y: 5.0,
                z: 5.0,
                size: 2.0,
                ..Default::default()
            }],
            edges: vec![],
        };
        let ds = LvDataset {
            source_path: None,
            sheets: vec![sheet_a, sheet_b],
            all_labels: vec!["appearing".into()],
        };
        // frame 0 => alpha=0.0, node is absent in T0 so fade-in from size=0
        let frame = compute_frame(&ds, &lis4(), 0);
        let inst = &frame.instances[0];
        assert!(
            (inst.size - 0.0).abs() < 1e-6,
            "fade-in at alpha=0 should have size 0"
        );
    }
}
