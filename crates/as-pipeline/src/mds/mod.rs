//! MDS module dispatcher.

pub mod classical;
pub mod force_directed;
pub mod landmark;
pub mod multilevel;
pub mod pivot;
pub mod smacof;
pub mod tsne;
pub mod umap;

use anyhow::{bail, Result};

use crate::error::AsError;
use crate::types::{DistanceMatrix, MdsConfig, MdsCoordinates, MdsDimMode};

/// Run MDS on a distance matrix, selecting the algorithm from `cfg`.
///
/// `MdsConfig::Auto` selects Classical MDS for n<800, PivotMds otherwise.
/// Pivot count scales as max(50, sqrt(n)), capped at 200.
pub fn run_mds(
    dist: &DistanceMatrix,
    cfg: &MdsConfig,
    dim_mode: MdsDimMode,
) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = resolve_dims(dim_mode, n);

    match cfg {
        MdsConfig::Auto => {
            if n < 800 {
                classical::classical_mds(dist, dims)
            } else {
                let n_pivots = auto_pivot_count(n);
                pivot::pivot_mds(dist, dims, n_pivots)
            }
        }
        MdsConfig::Classical => classical::classical_mds(dist, dims),
        MdsConfig::Smacof(smacof_cfg) => smacof::smacof(dist, dims, smacof_cfg),
        MdsConfig::PivotMds { n_pivots } => pivot::pivot_mds(dist, dims, *n_pivots),
        MdsConfig::Multilevel {
            levels,
            refine_iters,
        } => multilevel::multilevel_mds(dist, dims, *levels, *refine_iters),
        MdsConfig::Landmark { n_landmarks } => landmark::landmark_mds(dist, dims, *n_landmarks),
    }
}

/// Auto-select pivot count: max(50, sqrt(n)), capped at 200.
fn auto_pivot_count(n: usize) -> usize {
    let sqrt_n = (n as f64).sqrt() as usize;
    sqrt_n.clamp(50, 200).min(n)
}

fn resolve_dims(mode: MdsDimMode, n: usize) -> usize {
    match mode {
        // `Visual` is the legacy API name for the 2D planar layout mode.
        MdsDimMode::Visual => 2,
        MdsDimMode::Fixed(d) => d.min(n - 1),
        MdsDimMode::Maximum => (n - 1).max(1),
    }
}
