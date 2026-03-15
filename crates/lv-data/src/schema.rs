use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::DataError;

// ─────────────────────────────────────────────────────────────────────────────
// ShapeKind
// ─────────────────────────────────────────────────────────────────────────────

/// The six renderable node shapes.
///
/// The discriminant value matches the `shape_id` field uploaded to the GPU
/// (see [`GpuInstance`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum ShapeKind {
    Sphere = 0,
    Point = 1,
    Torus = 2,
    Pyramid = 3,
    Cube = 4,
    Cylinder = 5,
}

impl ShapeKind {
    /// GPU shape identifier — matches the `#repr(u32)` discriminant.
    #[inline]
    pub fn gpu_id(self) -> u32 {
        self as u32
    }

    /// Whether this shape supports independent spin on each axis.
    #[inline]
    pub fn supports_spin(self) -> bool {
        matches!(
            self,
            ShapeKind::Torus | ShapeKind::Pyramid | ShapeKind::Cube | ShapeKind::Cylinder
        )
    }

    /// Human-readable description of what the `size_alpha` field controls for
    /// this shape (informational; used in GUI tooltips).
    pub fn size_alpha_meaning(self) -> &'static str {
        match self {
            ShapeKind::Torus => "inner hole fraction [0, 1)",
            ShapeKind::Pyramid => "height multiplier",
            ShapeKind::Cube => "depth/width ratio",
            ShapeKind::Cylinder => "height multiplier",
            ShapeKind::Sphere => "unused",
            ShapeKind::Point => "unused",
        }
    }

    /// All variants in GPU-ID order — useful for iteration.
    pub const ALL: [ShapeKind; 6] = [
        ShapeKind::Sphere,
        ShapeKind::Point,
        ShapeKind::Torus,
        ShapeKind::Pyramid,
        ShapeKind::Cube,
        ShapeKind::Cylinder,
    ];
}

impl fmt::Display for ShapeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ShapeKind::Sphere => "sphere",
            ShapeKind::Point => "point",
            ShapeKind::Torus => "torus",
            ShapeKind::Pyramid => "pyramid",
            ShapeKind::Cube => "cube",
            ShapeKind::Cylinder => "cylinder",
        };
        write!(f, "{s}")
    }
}

impl FromStr for ShapeKind {
    type Err = DataError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "sphere" => Ok(ShapeKind::Sphere),
            "point" => Ok(ShapeKind::Point),
            "torus" => Ok(ShapeKind::Torus),
            "pyramid" => Ok(ShapeKind::Pyramid),
            "cube" => Ok(ShapeKind::Cube),
            "cylinder" => Ok(ShapeKind::Cylinder),
            other => Err(DataError::UnknownShape(other.to_owned())),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EtvRow — one node / time-point row (19 fixed columns)
// ─────────────────────────────────────────────────────────────────────────────

/// A single node row from an ETV XLSX file.
///
/// Column positions (0-indexed) are fixed; column headers in the file are
/// ignored.  All coordinates are in the normalised visualisation space that
/// `as-pipeline` outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtvRow {
    /// Col 0 — non-empty node label, max 255 chars.
    pub label: String,
    /// Col 1-3 — spatial coordinates.
    pub x: f64,
    pub y: f64,
    pub z: f64,
    /// Col 4 — rendered size; must be > 0.
    pub size: f64,
    /// Col 5 — shape-specific secondary size parameter; must be >= 0.
    pub size_alpha: f64,
    /// Col 6-8 — per-axis spin speed in degrees/frame.
    pub spin_x: f64,
    pub spin_y: f64,
    pub spin_z: f64,
    /// Col 9 — shape variant.
    pub shape: ShapeKind,
    /// Col 10-12 — RGB colour in [0, 1].
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    /// Col 13 — MIDI note [0, 127].
    pub note: u8,
    /// Col 14 — MIDI instrument; [0, 127] standard, [128, 365] extended.
    pub instrument: u16,
    /// Col 15 — MIDI channel [0, 15].
    pub channel: u8,
    /// Col 16 — MIDI velocity [1, 127].
    pub velocity: u8,
    /// Col 17 — cluster membership value; >= 0.
    pub cluster_value: f64,
    /// Col 18 — LIS beat count; the LIS value must be >= 2 * beats.
    pub beats: u32,
}

impl Default for EtvRow {
    fn default() -> Self {
        EtvRow {
            label: String::new(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            size: 1.0,
            size_alpha: 0.0,
            spin_x: 0.0,
            spin_y: 0.0,
            spin_z: 0.0,
            shape: ShapeKind::Sphere,
            color_r: 1.0,
            color_g: 1.0,
            color_b: 1.0,
            note: 60,
            instrument: 0,
            channel: 0,
            velocity: 64,
            cluster_value: 0.0,
            beats: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EdgeRow
// ─────────────────────────────────────────────────────────────────────────────

/// A directed or undirected edge between two labelled nodes.
///
/// The edge section in the XLSX begins at the first row where column 0
/// contains the literal string `"from"` (case-insensitive).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeRow {
    /// Label of the source node — must match a label in the same sheet.
    pub from: String,
    /// Label of the target node — must match a label in the same sheet.
    pub to: String,
    /// Edge weight / similarity strength.
    pub strength: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// EtvSheet
// ─────────────────────────────────────────────────────────────────────────────

/// One worksheet from an ETV workbook.
///
/// Each sheet represents one *time point* in the LIS animation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtvSheet {
    /// Worksheet name as reported by calamine.
    pub name: String,
    /// Zero-based sheet index within the workbook.
    pub sheet_index: usize,
    /// Node rows (object section).
    pub rows: Vec<EtvRow>,
    /// Edge rows (edge section, may be empty).
    pub edges: Vec<EdgeRow>,
}

// ─────────────────────────────────────────────────────────────────────────────
// EtvDataset
// ─────────────────────────────────────────────────────────────────────────────

/// The full contents of an ETV workbook — one or more time-point sheets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EtvDataset {
    /// Original file path, if loaded from disk.
    pub source_path: Option<PathBuf>,
    /// Sheets in workbook order.
    pub sheets: Vec<EtvSheet>,
    /// Deduplicated union of all labels across all sheets, in first-seen order.
    pub all_labels: Vec<String>,
}

impl EtvDataset {
    /// Number of time points (= number of sheets).
    pub fn time_points(&self) -> usize {
        self.sheets.len()
    }

    /// Maximum number of objects across all sheets.
    pub fn max_objects(&self) -> usize {
        self.sheets.iter().map(|s| s.rows.len()).max().unwrap_or(0)
    }

    /// Estimated size in bytes of a fully pre-computed LIS frame buffer.
    ///
    /// `frames = (time_points - 1) * lis_value`
    /// `bytes  = frames * max_objects * size_of::<GpuInstance>()`
    pub fn estimated_lis_buffer_bytes(&self, lis_value: u32) -> usize {
        let time_points = self.time_points();
        if time_points < 2 {
            return 0;
        }
        let frames = (time_points - 1) * lis_value as usize;
        frames * self.max_objects() * std::mem::size_of::<GpuInstance>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuInstance — per-instance data uploaded to the GPU
// ─────────────────────────────────────────────────────────────────────────────

/// Per-instance data for one rendered node.
///
/// Layout is `repr(C)` and implements [`bytemuck::Pod`] so it can be uploaded
/// directly to a `wgpu` vertex/instance buffer with zero copies.
///
/// Size: `4 * 16 = 64 bytes` (16-byte aligned; fits cleanly into GPU buffers).
///
/// Field layout:
/// ```text
///  offset  size  field
///  0        12   position [f32; 3]
///  12        4   size     f32
///  16        4   size_alpha f32
///  20       12   _pad0    [f32; 3]   ← keeps color 16-byte aligned
///  32       16   color    [f32; 4]
///  48       12   spin     [f32; 3]
///  60        4   shape_id u32
/// total    64
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable, Serialize, Deserialize)]
pub struct GpuInstance {
    /// World-space position (x, y, z).
    pub position: [f32; 3],
    /// Uniform scale.
    pub size: f32,
    /// Shape-specific secondary scale (see [`ShapeKind::size_alpha_meaning`]).
    pub size_alpha: f32,
    /// Padding — keeps `color` at a 16-byte offset.
    pub _pad0: [f32; 3],
    /// RGBA colour.
    pub color: [f32; 4],
    /// Per-axis spin speed (degrees/frame) — only used when
    /// [`ShapeKind::supports_spin`] is true.
    pub spin: [f32; 3],
    /// GPU shape selector — see [`ShapeKind::gpu_id`].
    pub shape_id: u32,
}

impl GpuInstance {
    /// Construct a [`GpuInstance`] from an [`EtvRow`] at a given position.
    ///
    /// `position` is provided separately because it comes from the MDS
    /// pipeline output, not directly from the row.
    pub fn from_row(row: &EtvRow, position: [f32; 3]) -> Self {
        GpuInstance {
            position,
            size: row.size as f32,
            size_alpha: row.size_alpha as f32,
            _pad0: [0.0; 3],
            color: [row.color_r, row.color_g, row.color_b, 1.0],
            spin: [row.spin_x as f32, row.spin_y as f32, row.spin_z as f32],
            shape_id: row.shape.gpu_id(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LIS types
// ─────────────────────────────────────────────────────────────────────────────

/// One interpolated animation frame between two adjacent time-point sheets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LisFrame {
    /// Interpolated GPU instances for every node active at this frame.
    pub instances: Vec<GpuInstance>,
    /// Index into `EtvDataset::sheets` of the *source* sheet for this segment.
    pub slice_index: u32,
    /// Absolute frame index across the whole animation.
    pub transition_index: u32,
    /// Frame index within the current sheet-to-sheet segment [0, lis).
    pub local_slice: u32,
}

/// A fully pre-computed (or streaming) LIS animation buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LisBuffer {
    /// All pre-computed frames (empty if `streaming == true`).
    pub frames: Vec<LisFrame>,
    /// When true, frames are generated on demand rather than stored here.
    pub streaming: bool,
    /// LIS value used to generate this buffer.
    pub lis: u32,
    /// Total number of animation frames.
    pub total_frames: u32,
}

/// Configuration for LIS (Linear Interpolation Sweep) animation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LisConfig {
    /// Number of interpolated frames between each pair of time points.
    /// Must be >= 2.  Default: 30.
    pub lis_value: u32,
    /// Optional target frame rate cap (Hz).  `None` = uncapped.
    pub target_fps: Option<u32>,
    /// Whether the animation loops at the end.  Default: true.
    pub looping: bool,
    /// Playback speed multiplier.  Default: 1.0.
    pub speed: f32,
}

impl Default for LisConfig {
    fn default() -> Self {
        LisConfig {
            lis_value: 30,
            target_fps: None,
            looping: true,
            speed: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_gpu_ids_are_unique() {
        let ids: Vec<u32> = ShapeKind::ALL.iter().map(|s| s.gpu_id()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(ids.len(), sorted.len(), "duplicate gpu_id detected");
    }

    #[test]
    fn shape_gpu_ids_are_zero_to_five() {
        let mut ids: Vec<u32> = ShapeKind::ALL.iter().map(|s| s.gpu_id()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn shape_from_str_case_insensitive() {
        assert_eq!("Sphere".parse::<ShapeKind>().unwrap(), ShapeKind::Sphere);
        assert_eq!("TORUS".parse::<ShapeKind>().unwrap(), ShapeKind::Torus);
        assert_eq!("  cube  ".parse::<ShapeKind>().unwrap(), ShapeKind::Cube);
        assert_eq!(
            "cylinder".parse::<ShapeKind>().unwrap(),
            ShapeKind::Cylinder
        );
        assert_eq!("PYRAMID".parse::<ShapeKind>().unwrap(), ShapeKind::Pyramid);
        assert_eq!("point".parse::<ShapeKind>().unwrap(), ShapeKind::Point);
    }

    #[test]
    fn shape_from_str_unknown_errors() {
        assert!("hexagon".parse::<ShapeKind>().is_err());
        assert!("".parse::<ShapeKind>().is_err());
    }

    #[test]
    fn shape_supports_spin() {
        assert!(!ShapeKind::Sphere.supports_spin());
        assert!(!ShapeKind::Point.supports_spin());
        assert!(ShapeKind::Torus.supports_spin());
        assert!(ShapeKind::Pyramid.supports_spin());
        assert!(ShapeKind::Cube.supports_spin());
        assert!(ShapeKind::Cylinder.supports_spin());
    }

    #[test]
    fn gpu_instance_size_is_64() {
        assert_eq!(
            std::mem::size_of::<GpuInstance>(),
            64,
            "GpuInstance must be exactly 64 bytes for GPU buffer alignment"
        );
    }

    #[test]
    fn gpu_instance_align_is_4() {
        assert_eq!(std::mem::align_of::<GpuInstance>(), 4);
    }

    #[test]
    fn lis_config_defaults() {
        let cfg = LisConfig::default();
        assert_eq!(cfg.lis_value, 30);
        assert!(cfg.looping);
        assert_eq!(cfg.speed, 1.0);
        assert!(cfg.target_fps.is_none());
    }

    #[test]
    fn etv_dataset_estimated_buffer_bytes() {
        // 3 sheets, 10 objects each, LIS=30 → (3-1)*30 = 60 frames
        // 60 * 10 * 64 = 38 400 bytes
        let sheet = EtvSheet {
            name: "t0".into(),
            sheet_index: 0,
            rows: (0..10).map(|_| EtvRow::default()).collect(),
            edges: vec![],
        };
        let dataset = EtvDataset {
            source_path: None,
            sheets: vec![sheet.clone(), sheet.clone(), sheet],
            all_labels: vec![],
        };
        assert_eq!(dataset.estimated_lis_buffer_bytes(30), 60 * 10 * 64);
    }
}
