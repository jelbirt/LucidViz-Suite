//! MDS module dispatcher.

pub mod classical;
pub mod pivot;
pub mod smacof;

use anyhow::Result;

use crate::types::{DistanceMatrix, MdsConfig, MdsCoordinates, MdsDimMode};

/// Run MDS on a distance matrix, selecting the algorithm from `cfg`.
///
/// `MdsConfig::Auto` selects Classical MDS for n<500, PivotMds otherwise.
pub fn run_mds(
    dist: &DistanceMatrix,
    cfg: &MdsConfig,
    dim_mode: MdsDimMode,
) -> Result<MdsCoordinates> {
    let n = dist.n;
    let dims = resolve_dims(dim_mode, n);

    match cfg {
        MdsConfig::Auto => {
            if n < 500 {
                classical::classical_mds(dist, dims)
            } else {
                let n_pivots = 50.min(n);
                pivot::pivot_mds(dist, dims, n_pivots)
            }
        }
        MdsConfig::Classical => classical::classical_mds(dist, dims),
        MdsConfig::Smacof(smacof_cfg) => smacof::smacof(dist, dims, smacof_cfg),
        MdsConfig::PivotMds { n_pivots } => pivot::pivot_mds(dist, dims, *n_pivots),
    }
}

fn resolve_dims(mode: MdsDimMode, n: usize) -> usize {
    match mode {
        // `Visual` is the legacy API name for the 2D planar layout mode.
        MdsDimMode::Visual => 2,
        MdsDimMode::Fixed(d) => d.min(n - 1),
        MdsDimMode::Maximum => (n - 1).max(1),
    }
}
