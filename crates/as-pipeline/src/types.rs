//! Core types for the AlignSpace pipeline.

use crate::error::AsError;
pub use lv_data::analysis::CentralityReport;
use lv_data::schema::LvDataset;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Structural equivalence matrix
// ---------------------------------------------------------------------------

/// A symmetric distance matrix used by MDS and related alignment stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceMatrix {
    pub labels: Vec<String>,
    /// Row-major, n×n.
    pub data: Vec<f64>,
    pub n: usize,
}

impl DistanceMatrix {
    pub fn new(labels: Vec<String>, data: Vec<f64>) -> Result<Self, AsError> {
        let n = labels.len();
        if data.len() != n * n {
            return Err(AsError::DimensionMismatch(format!(
                "DistanceMatrix expected {} values for {} labels, got {}",
                n * n,
                n,
                data.len()
            )));
        }
        for i in 0..n {
            for j in 0..n {
                let value = data[i * n + j];
                if !value.is_finite() {
                    return Err(AsError::InvalidMatrix(format!(
                        "distance[{i},{j}] is not finite"
                    )));
                }
                if value < 0.0 {
                    return Err(AsError::InvalidMatrix(format!(
                        "distance[{i},{j}] is negative: {value}"
                    )));
                }
                if i == j && value.abs() > 1e-12 {
                    return Err(AsError::InvalidMatrix(format!(
                        "distance diagonal at [{i},{j}] must be 0, got {value}"
                    )));
                }
                if j > i {
                    let other = data[j * n + i];
                    if (value - other).abs() > 1e-12 {
                        return Err(AsError::InvalidMatrix(format!(
                            "distance matrix is not symmetric at [{i},{j}] ({value}) and [{j},{i}] ({other})"
                        )));
                    }
                }
            }
        }
        Ok(DistanceMatrix { labels, data, n })
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }
}

/// Legacy alias retained for compatibility with existing code and serialized
/// artifacts that still refer to a structural-equivalence matrix.
pub type SeMatrix = DistanceMatrix;

// ---------------------------------------------------------------------------
// MDS output
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MdsCoordinates {
    pub labels: Vec<String>,
    /// Row-major, n×dims.
    pub data: Vec<f64>,
    pub n: usize,
    pub dims: usize,
    pub stress: f64,
    pub algorithm: MdsAlgorithm,
}

impl MdsCoordinates {
    pub fn new(
        labels: Vec<String>,
        data: Vec<f64>,
        dims: usize,
        stress: f64,
        algorithm: MdsAlgorithm,
    ) -> Result<Self, AsError> {
        let n = labels.len();
        if data.len() != n * dims {
            return Err(AsError::DimensionMismatch(format!(
                "MdsCoordinates expected {} values for {} labels across {} dims, got {}",
                n * dims,
                n,
                dims,
                data.len()
            )));
        }
        Ok(MdsCoordinates {
            labels,
            data,
            n,
            dims,
            stress,
            algorithm,
        })
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.dims..(i + 1) * self.dims]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MdsAlgorithm {
    Classical,
    Smacof,
    PivotMds,
}

// ---------------------------------------------------------------------------
// Procrustes result
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcrustesResult {
    pub aligned: MdsCoordinates,
    /// Flattened rotation matrix (dims×dims, row-major).
    pub rotation: Vec<f64>,
    pub scale: f64,
    pub translation: Vec<f64>,
    pub residual: f64,
}

// ---------------------------------------------------------------------------
// Centrality
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentralityMode {
    UndirectedLegacy,
    Directed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CentralityState {
    Computed(CentralityReport),
    Unavailable { labels: Vec<String>, reason: String },
}

// ---------------------------------------------------------------------------
// MDS configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MdsConfig {
    /// Auto-select: Classical for n<500, PivotMds otherwise.
    Auto,
    Classical,
    Smacof(SmacofConfig),
    PivotMds {
        n_pivots: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmacofConfig {
    pub max_iter: u32,
    pub tolerance: f64,
    pub init: SmacofInit,
}

impl Default for SmacofConfig {
    fn default() -> Self {
        SmacofConfig {
            max_iter: 300,
            tolerance: 1e-6,
            init: SmacofInit::Classical,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmacofInit {
    Classical,
    Random(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MdsDimMode {
    /// Use all valid dimensions.
    Maximum,
    /// Legacy public name for the 2D planar layout mode.
    Visual,
    /// Use exactly this many dimensions.
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcrustesMode {
    /// Align each time-step to the previous (time series).
    TimeSeries,
    /// Align each time-step to the first slice to avoid chain drift.
    TimeSeriesAnchored,
    /// Find the pair with best alignment and propagate.
    OptimalChoice,
    /// Generalized Procrustes Analysis: iteratively align all configurations
    /// to their consensus (mean) until convergence. Eliminates both chain
    /// drift and anchor bias.
    GPA,
    /// Skip Procrustes entirely.
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// Scale each time-step independently to fit the target range.
    Independent,
    /// Scale the whole series with one shared factor.
    Global,
}

// ---------------------------------------------------------------------------
// Pipeline I/O
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AsPipelineInput {
    /// (dataset_name, adjacency_matrix) pairs — one per time step.
    pub datasets: Vec<(String, Array2<f64>)>,
    pub labels: Vec<String>,
    pub mds_config: MdsConfig,
    pub procrustes_mode: ProcrustesMode,
    pub mds_dims: MdsDimMode,
    pub normalize: bool,
    pub normalization_mode: NormalizationMode,
    pub target_range: f64,
    pub procrustes_scale: bool,
    pub centrality_mode: CentralityMode,
}

#[derive(Debug, Clone)]
pub struct AsDistancePipelineInput {
    /// (dataset_name, distance_matrix) pairs — one per time step.
    pub datasets: Vec<(String, DistanceMatrix)>,
    pub mds_config: MdsConfig,
    pub procrustes_mode: ProcrustesMode,
    pub mds_dims: MdsDimMode,
    pub normalize: bool,
    pub normalization_mode: NormalizationMode,
    pub target_range: f64,
    pub procrustes_scale: bool,
    pub centrality_mode: CentralityMode,
}

#[derive(Debug)]
pub struct AsPipelineResult {
    pub coordinates: Vec<MdsCoordinates>,
    pub procrustes: Vec<ProcrustesResult>,
    pub centralities: Vec<CentralityState>,
    pub centrality_mode: CentralityMode,
    pub distance_matrices: Vec<DistanceMatrix>,
    pub lv_dataset: LvDataset,
}
