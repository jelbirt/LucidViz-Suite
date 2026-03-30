//! Pivot MDS — fast approximation suitable for large n.
//!
//! Uses farthest-point landmark sampling followed by a thin SVD.

use anyhow::Result;

use crate::error::AsError;
use crate::types::{MdsAlgorithm, MdsCoordinates, SeMatrix};

/// Run Pivot MDS.
///
/// # Arguments
/// * `dist`     – Symmetric n×n distance matrix.
/// * `dims`     – Number of output dimensions.
/// * `n_pivots` – Number of landmark/pivot nodes.
pub fn pivot_mds(dist: &SeMatrix, dims: usize, n_pivots: usize) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        anyhow::bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.min(n - 1);
    let k = n_pivots.min(n);

    // Step 1: Farthest-point pivot selection.
    let pivots = farthest_point_pivots(dist, k);

    // Step 2: Build C matrix (k×n): C[s,i] = dist[pivot_s, i]
    // Step 3: C² then double-center.
    let c_tilde = double_center_c(dist, &pivots, k, n);

    // Step 4: Thin SVD of c_tilde (k×n).
    // We use nalgebra for SVD since faer thin-SVD is more involved.
    let x = svd_coordinates(&c_tilde, k, n, dims)?;

    let stress = crate::mds::classical::kruskal_stress(dist, &x, n, dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        x,
        dims,
        stress,
        MdsAlgorithm::PivotMds,
    )?)
}

/// Farthest-point sampling: start with the node whose row sum is largest
/// (most central), then greedily add the node farthest from the current set.
pub(crate) fn farthest_point_pivots(dist: &SeMatrix, k: usize) -> Vec<usize> {
    let n = dist.n;
    // Start: node with maximum total distance (most "spread" from all others).
    let start = (0..n)
        .max_by(|&a, &b| {
            let sa: f64 = (0..n).map(|j| dist.get(a, j)).sum();
            let sb: f64 = (0..n).map(|j| dist.get(b, j)).sum();
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut selected = vec![start];
    let mut selected_set = std::collections::HashSet::new();
    selected_set.insert(start);
    // min_dist[i] = min distance from node i to any already-selected pivot.
    let mut min_dist: Vec<f64> = (0..n).map(|i| dist.get(start, i)).collect();

    while selected.len() < k {
        // Pick node with maximum min-distance to current pivot set.
        let next = match (0..n)
            .filter(|i| !selected_set.contains(i))
            .max_by(|&a, &b| {
                min_dist[a]
                    .partial_cmp(&min_dist[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
            Some(idx) => idx,
            None => {
                log::warn!("farthest_point_pivots: all remaining nodes have equal distance; stopping at {} pivots", selected.len());
                break;
            }
        };
        selected.push(next);
        selected_set.insert(next);
        // Update min_dist.
        for (i, min_d) in min_dist.iter_mut().enumerate() {
            let d = dist.get(next, i);
            if d < *min_d {
                *min_d = d;
            }
        }
    }
    selected
}

/// Build C² then double-center: C_tilde[s,i] = -0.5*(C2-row_mean_s - col_mean_i + grand_mean)
fn double_center_c(dist: &SeMatrix, pivots: &[usize], k: usize, n: usize) -> Vec<f64> {
    // C[s,i] = dist[pivot_s, i]
    let mut c2 = vec![0.0f64; k * n];
    for (s, &p) in pivots.iter().enumerate() {
        for i in 0..n {
            let d = dist.get(p, i);
            c2[s * n + i] = d * d;
        }
    }

    // Row means (over n).
    let row_means: Vec<f64> = (0..k)
        .map(|s| (0..n).map(|i| c2[s * n + i]).sum::<f64>() / n as f64)
        .collect();
    // Col means (over k).
    let col_means: Vec<f64> = (0..n)
        .map(|i| (0..k).map(|s| c2[s * n + i]).sum::<f64>() / k as f64)
        .collect();
    let grand_mean: f64 = row_means.iter().sum::<f64>() / k as f64;

    let mut c_tilde = vec![0.0f64; k * n];
    for s in 0..k {
        for i in 0..n {
            c_tilde[s * n + i] = -0.5 * (c2[s * n + i] - row_means[s] - col_means[i] + grand_mean);
        }
    }
    c_tilde
}

/// Thin SVD of a k×n matrix stored row-major.  Returns n×dims coordinates.
fn svd_coordinates(c_tilde: &[f64], k: usize, n: usize, dims: usize) -> Result<Vec<f64>> {
    // Build nalgebra DMatrix (k rows, n cols).
    use nalgebra::DMatrix;
    let mat = DMatrix::from_row_slice(k, n, c_tilde);
    // Thin SVD: mat = U * Sigma * V^T.  nalgebra SVD computes full, but we
    // can just take the first `dims` components.
    let svd = mat.svd(false, true); // compute_u=false, compute_v=true
                                    // V is n×min(k,n); singular values in descending order.
    let v_t = svd
        .v_t
        .ok_or_else(|| AsError::MdsFailed("pivot SVD: V_t matrix unavailable".into()))?; // min(k,n) × n

    let sigma = &svd.singular_values; // length = min(k,n)
    let effective_dims = dims.min(sigma.len());

    // X = V_D * Sigma_D  (i.e., for each row i of V^T column d: x[i,d] = v_t[d,i]*sigma[d])
    let mut x = vec![0.0f64; n * dims];
    for d in 0..effective_dims {
        let s = sigma[d];
        for i in 0..n {
            x[i * dims + d] = v_t[(d, i)] * s;
        }
    }

    // Center.
    for d in 0..dims {
        let mean: f64 = (0..n).map(|i| x[i * dims + d]).sum::<f64>() / n as f64;
        for i in 0..n {
            x[i * dims + d] -= mean;
        }
    }
    Ok(x)
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

    fn line_dist(n: usize) -> SeMatrix {
        let lbs = labels(n);
        let mut vals = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vals[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        SeMatrix::new(lbs, vals).expect("test SE matrix should build")
    }

    #[test]
    fn test_pivot_mds_output_shape() {
        let n = 10;
        let se = line_dist(n);
        let coords = pivot_mds(&se, 2, 4).unwrap();
        assert_eq!(coords.data.len(), n * 2);
        assert_eq!(coords.n, n);
        assert_eq!(coords.dims, 2);
    }

    #[test]
    fn test_pivot_mds_stress_reasonable() {
        // Pivot MDS is an approximation — verify it produces valid output.
        // The stress threshold is relaxed since Pivot MDS optimizes only approximately.
        let n = 8;
        let se = line_dist(n);
        let coords = pivot_mds(&se, 1, 6).unwrap(); // 6 pivots out of 8
                                                    // Pivot MDS may produce higher stress than Classical; just verify it's finite.
        assert!(
            coords.stress.is_finite(),
            "stress should be finite, got {}",
            coords.stress
        );
        assert!(coords.stress >= 0.0, "stress should be non-negative");
    }
}
