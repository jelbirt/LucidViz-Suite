//! Multilevel MDS — coarsen → base solve → interpolate → refine.
//!
//! Based on the Brandes & Pich (2006) multilevel approach:
//! 1. Build a hierarchy of point subsets via farthest-point sampling,
//!    halving at each level until the coarsest level has ~50-100 points.
//! 2. Solve the coarsest level with Classical MDS.
//! 3. At each finer level, interpolate new points via distance-weighted
//!    averaging of their k nearest already-placed neighbours, then run
//!    a few SMACOF/Guttman refinement iterations.
//! 4. Return the final coordinates with Kruskal stress.

use anyhow::Result;
use rayon::prelude::*;

use crate::error::AsError;
use crate::mds::classical::{classical_mds, kruskal_stress};
use crate::mds::pivot::farthest_point_pivots;
use crate::mds::smacof::{center_coords, guttman_step};
use crate::types::{DistanceMatrix, MdsAlgorithm, MdsCoordinates};

/// Minimum size for the coarsest level before we just use Classical MDS.
const MIN_COARSEST: usize = 30;

/// Number of nearest neighbours used for interpolation.
const INTERP_K: usize = 5;

/// Run multilevel MDS on a distance matrix.
///
/// # Arguments
/// * `dist`         – Symmetric n×n distance matrix.
/// * `dims`         – Number of output dimensions.
/// * `levels`       – Number of coarsening levels (typically 3-5).
/// * `refine_iters` – SMACOF/Guttman iterations per level during
///   prolongation.
pub fn multilevel_mds(
    dist: &DistanceMatrix,
    dims: usize,
    levels: usize,
    refine_iters: u32,
) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        anyhow::bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.min(n - 1);

    // --- Phase 1: Build coarsening hierarchy ---
    let hierarchy = build_hierarchy(dist, n, levels);
    log::debug!(
        "multilevel MDS: {} levels, sizes: {:?}",
        hierarchy.len(),
        hierarchy.iter().map(|h| h.len()).collect::<Vec<_>>()
    );

    // --- Phase 2: Solve coarsest level with Classical MDS ---
    let coarsest = &hierarchy[hierarchy.len() - 1];
    let sub_dist = extract_sub_distance(dist, coarsest);
    let base = classical_mds(&sub_dist, dims)?;

    // Place coarsest-level coordinates into a full n×dims buffer.
    // `placed[i]` tracks whether node i has coordinates.
    let mut coords = vec![0.0f64; n * dims];
    let mut placed = vec![false; n];
    for (local, &global) in coarsest.iter().enumerate() {
        for d in 0..dims {
            coords[global * dims + d] = base.data[local * dims + d];
        }
        placed[global] = true;
    }

    // --- Phase 3: Prolongation — from coarsest to finest ---
    // Walk levels from second-coarsest back to the full set (level 0).
    for level_idx in (0..hierarchy.len() - 1).rev() {
        let level_set = &hierarchy[level_idx];

        // Identify new points at this level that are not yet placed.
        let new_points: Vec<usize> = level_set
            .iter()
            .copied()
            .filter(|&idx| !placed[idx])
            .collect();

        // Interpolate positions for new points.
        interpolate_new_points(dist, &mut coords, &placed, &new_points, dims);
        for &idx in &new_points {
            placed[idx] = true;
        }

        // Build the sub-distance-matrix for this level's points and
        // run Guttman refinement iterations.
        let sub_dist_level = extract_sub_distance(dist, level_set);
        let mut sub_coords = extract_sub_coords(&coords, level_set, dims);

        let mut prev_stress = kruskal_stress(&sub_dist_level, &sub_coords, level_set.len(), dims);
        for iter in 0..refine_iters {
            let new_sub = guttman_step(&sub_dist_level, &sub_coords, level_set.len(), dims);
            sub_coords = new_sub;
            center_coords(&mut sub_coords, level_set.len(), dims);

            let cur_stress = kruskal_stress(&sub_dist_level, &sub_coords, level_set.len(), dims);
            if iter > 0 && (prev_stress - cur_stress).abs() < 1e-6 {
                log::trace!("multilevel level {} converged at iter {}", level_idx, iter);
                break;
            }
            prev_stress = cur_stress;
        }

        // Write refined sub-coordinates back into the full buffer.
        for (local, &global) in level_set.iter().enumerate() {
            for d in 0..dims {
                coords[global * dims + d] = sub_coords[local * dims + d];
            }
        }
    }

    // --- Phase 4: Final stress computation ---
    let stress = kruskal_stress(dist, &coords, n, dims);
    log::debug!("multilevel MDS final stress: {stress:.6}");

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        coords,
        dims,
        stress,
        MdsAlgorithm::Multilevel,
    )?)
}

/// Build the coarsening hierarchy.
///
/// `hierarchy[0]` = all n indices (finest), `hierarchy[last]` = coarsest.
/// Each level keeps roughly half the points of the previous one, down to
/// at least `MIN_COARSEST`.
fn build_hierarchy(dist: &DistanceMatrix, n: usize, levels: usize) -> Vec<Vec<usize>> {
    let mut hierarchy = Vec::with_capacity(levels + 1);

    // Level 0: all points.
    let all: Vec<usize> = (0..n).collect();
    hierarchy.push(all);

    for _ in 1..=levels {
        let prev = hierarchy.last().unwrap();
        let prev_n = prev.len();

        // Stop coarsening if already small enough.
        if prev_n <= MIN_COARSEST {
            break;
        }

        // Target size: half, but at least MIN_COARSEST.
        let target = (prev_n / 2).max(MIN_COARSEST);

        // Use farthest-point sampling on the full distance matrix to
        // pick `target` representatives from the previous level's set.
        // We build a sub-distance-matrix for the previous level and
        // sample from that.
        let sub_dist = extract_sub_distance(dist, prev);
        let local_pivots = farthest_point_pivots(&sub_dist, target);

        // Map local pivot indices back to global indices.
        let global_pivots: Vec<usize> = local_pivots.iter().map(|&li| prev[li]).collect();

        hierarchy.push(global_pivots);
    }

    hierarchy
}

/// Extract a sub-distance-matrix for the given subset of global indices.
fn extract_sub_distance(dist: &DistanceMatrix, indices: &[usize]) -> DistanceMatrix {
    let m = indices.len();
    let mut data = vec![0.0f64; m * m];
    for (li, &gi) in indices.iter().enumerate() {
        for (lj, &gj) in indices.iter().enumerate() {
            data[li * m + lj] = dist.get(gi, gj);
        }
    }
    let labels: Vec<String> = indices.iter().map(|&i| dist.labels[i].clone()).collect();
    // Safety: the sub-matrix inherits symmetry and zero-diagonal from
    // the parent, so `new` cannot fail.
    DistanceMatrix::new(labels, data).expect("sub-distance matrix must be valid")
}

/// Extract sub-coordinates for the given subset of global indices.
fn extract_sub_coords(coords: &[f64], indices: &[usize], dims: usize) -> Vec<f64> {
    let mut sub = vec![0.0f64; indices.len() * dims];
    for (li, &gi) in indices.iter().enumerate() {
        for d in 0..dims {
            sub[li * dims + d] = coords[gi * dims + d];
        }
    }
    sub
}

/// Interpolate positions for new (unplaced) points using distance-weighted
/// averaging of their k nearest already-placed neighbours.
fn interpolate_new_points(
    dist: &DistanceMatrix,
    coords: &mut [f64],
    placed: &[bool],
    new_points: &[usize],
    dims: usize,
) {
    let n = dist.n;

    // Collect placed indices once.
    let placed_indices: Vec<usize> = (0..n).filter(|&i| placed[i]).collect();

    // Interpolate each new point (parallelised).
    let interpolated: Vec<(usize, Vec<f64>)> = new_points
        .par_iter()
        .map(|&p| {
            let k = INTERP_K.min(placed_indices.len());

            // Find k nearest placed neighbours by original distance.
            let mut neighbours: Vec<(usize, f64)> = placed_indices
                .iter()
                .map(|&q| (q, dist.get(p, q)))
                .collect();
            neighbours.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            neighbours.truncate(k);

            // Distance-weighted average (inverse-distance weighting).
            let mut pos = vec![0.0f64; dims];
            let mut weight_sum = 0.0f64;

            for &(q, d_pq) in &neighbours {
                let w = if d_pq < 1e-15 {
                    1e15 // essentially snap to this neighbour
                } else {
                    1.0 / d_pq
                };
                weight_sum += w;
                for d in 0..dims {
                    pos[d] += w * coords[q * dims + d];
                }
            }

            if weight_sum > 0.0 {
                for val in pos.iter_mut() {
                    *val /= weight_sum;
                }
            }

            (p, pos)
        })
        .collect();

    // Write interpolated positions back.
    for (idx, pos) in interpolated {
        for d in 0..dims {
            coords[idx * dims + d] = pos[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{}", i)).collect()
    }

    fn line_dist(n: usize) -> DistanceMatrix {
        let lbs = labels(n);
        let mut vals = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vals[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        DistanceMatrix::new(lbs, vals).expect("test distance matrix")
    }

    #[test]
    fn test_multilevel_small_matrix_shape() {
        let n = 10;
        let dist = line_dist(n);
        let coords = multilevel_mds(&dist, 2, 4, 20).unwrap();
        assert_eq!(coords.n, n);
        assert_eq!(coords.dims, 2);
        assert_eq!(coords.data.len(), n * 2);
        assert!(coords.stress.is_finite());
        assert_eq!(coords.labels.len(), n);
        assert_eq!(coords.algorithm, MdsAlgorithm::Multilevel);
    }

    #[test]
    fn test_multilevel_medium_stress_quality() {
        // Multilevel stress should be within 2x of classical on the same
        // input.
        let n = 100;
        let dist = line_dist(n);
        let ml = multilevel_mds(&dist, 3, 4, 20).unwrap();
        let cl = crate::mds::classical::classical_mds(&dist, 3).unwrap();
        assert!(
            ml.stress <= cl.stress * 2.0 + 0.01,
            "multilevel stress {} too high vs classical {}",
            ml.stress,
            cl.stress,
        );
    }

    #[test]
    fn test_coarsening_levels() {
        // n=1000 with 4 levels should produce roughly
        // 1000 -> 500 -> 250 -> 125 -> ~63
        let n = 1000;
        let dist = line_dist(n);
        let hierarchy = build_hierarchy(&dist, n, 4);
        assert_eq!(hierarchy[0].len(), 1000);
        assert!(hierarchy.len() >= 3, "expected at least 3 levels");
        // Each level should be roughly half the previous.
        for i in 1..hierarchy.len() {
            let ratio = hierarchy[i].len() as f64 / hierarchy[i - 1].len() as f64;
            assert!(
                ratio >= 0.3 && ratio <= 0.7,
                "level {} ratio {ratio} out of range (sizes: {} -> {})",
                i,
                hierarchy[i - 1].len(),
                hierarchy[i].len(),
            );
        }
    }

    #[test]
    fn test_single_level_fallback() {
        // When n is very small, multilevel should gracefully fall back
        // (hierarchy collapses to 1-2 levels) and still produce valid
        // output.
        let n = 5;
        let dist = line_dist(n);
        let coords = multilevel_mds(&dist, 2, 4, 20).unwrap();
        assert_eq!(coords.n, n);
        assert!(coords.stress.is_finite());
        assert!(coords.stress >= 0.0);
    }
}
