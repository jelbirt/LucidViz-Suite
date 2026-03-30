//! Classical (metric) MDS via eigendecomposition of the double-centred
//! distance matrix.
//!
//! Uses ndarray for the double-centring step and nalgebra for eigendecomposition.

use anyhow::{bail, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;

use crate::error::AsError;
use crate::types::{MdsAlgorithm, MdsCoordinates, SeMatrix};

/// Compute Classical MDS coordinates from a distance matrix.
///
/// # Arguments
/// * `dist`  – Symmetric n×n distance matrix.
/// * `dims`  – Number of output dimensions.
///
/// Returns `MdsCoordinates` with Kruskal stress-1.
pub fn classical_mds(dist: &SeMatrix, dims: usize) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.min(n - 1);

    // Step 1-2: Build D² and double-center directly into nalgebra DMatrix.
    // Avoids the ndarray→nalgebra element-by-element copy.
    let mut row_means = vec![0.0f64; n];
    let mut col_means = vec![0.0f64; n];
    let mut grand_sum = 0.0f64;
    let d2: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let d = dist.get(i, j);
            let d2_val = d * d;
            row_means[i] += d2_val;
            col_means[j] += d2_val;
            grand_sum += d2_val;
            d2_val
        })
        .collect();
    let n_f = n as f64;
    for v in &mut row_means {
        *v /= n_f;
    }
    for v in &mut col_means {
        *v /= n_f;
    }
    let grand_mean = grand_sum / (n_f * n_f);

    let b = DMatrix::<f64>::from_fn(n, n, |i, j| {
        -0.5 * (d2[i * n + j] - row_means[i] - col_means[j] + grand_mean)
    });

    // Step 3: Eigendecomposition of B (symmetric positive semi-definite).
    let eig = SymmetricEigen::new(b);
    let mut eigenpairs: Vec<(f64, usize)> = eig
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (value, idx))
        .collect();
    eigenpairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Step 4: Take top `dims` eigenvalues (descending), clamp negatives to 0.
    let mut coords_data = vec![0.0f64; n * dims];
    for (d, &(lam, idx)) in eigenpairs.iter().take(dims).enumerate() {
        let lam = lam.max(0.0);
        let scale = lam.sqrt();
        for i in 0..n {
            coords_data[i * dims + d] = eig.eigenvectors[(i, idx)] * scale;
        }
    }

    // Step 5: Kruskal stress-1.
    let stress = kruskal_stress(dist, &coords_data, n, dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        coords_data,
        dims,
        stress,
        MdsAlgorithm::Classical,
    )?)
}

/// Kruskal stress-1: sqrt( sum_{i<j}(d_hat - d)^2 / sum_{i<j} d^2 )
/// Parallelized over rows via rayon for large n.
pub(crate) fn kruskal_stress(dist: &SeMatrix, coords: &[f64], n: usize, dims: usize) -> f64 {
    let (num, denom) = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row_num = 0.0f64;
            let mut row_denom = 0.0f64;
            for j in (i + 1)..n {
                let d = dist.get(i, j);
                let d_hat = euclidean_dist(coords, i, j, dims);
                row_num += (d_hat - d) * (d_hat - d);
                row_denom += d * d;
            }
            (row_num, row_denom)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

    if denom < 1e-15 {
        0.0
    } else {
        (num / denom).sqrt()
    }
}

/// Euclidean distance between row i and row j of a row-major n×dims matrix.
pub(crate) fn euclidean_dist(data: &[f64], i: usize, j: usize, dims: usize) -> f64 {
    let mut sum = 0.0f64;
    for d in 0..dims {
        let diff = data[i * dims + d] - data[j * dims + d];
        sum += diff * diff;
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_se(n: usize, vals: Vec<f64>) -> SeMatrix {
        let labels: Vec<String> = (0..n).map(|i| format!("n{}", i)).collect();
        SeMatrix::new(labels, vals).expect("test SE matrix should build")
    }

    fn line_dist(n: usize) -> SeMatrix {
        let mut vals = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vals[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        make_se(n, vals)
    }

    #[test]
    fn test_classical_mds_known_input() {
        // 4 nodes on a 1-D line: 0, 1, 2, 3
        let se = line_dist(4);
        let coords = classical_mds(&se, 1).unwrap();
        assert_eq!(coords.n, 4);
        assert_eq!(coords.dims, 1);
        // Stress should be very small for a perfectly 1-D dataset.
        assert!(coords.stress < 0.05, "stress={} too high", coords.stress);
    }

    #[test]
    fn test_classical_mds_stress_metric() {
        // 3 equidistant nodes (equilateral triangle).
        let d = 1.0f64;
        let vals = vec![0.0, d, d, d, 0.0, d, d, d, 0.0];
        let se = make_se(3, vals);
        let coords = classical_mds(&se, 2).unwrap();
        assert!(
            coords.stress < 1e-6,
            "stress={} for perfect triangle",
            coords.stress
        );
    }

    #[test]
    fn test_classical_mds_output_shape() {
        let se = line_dist(5);
        let coords = classical_mds(&se, 3).unwrap();
        assert_eq!(coords.data.len(), 5 * 3);
    }

    #[test]
    fn test_mds_preserves_distance_ordering() {
        // Two nodes at distance 1, two at distance 5 — MDS should preserve
        // that the far pair is farther than the close pair in the embedding.
        let vals = vec![0.0, 1.0, 5.0, 1.0, 0.0, 4.0, 5.0, 4.0, 0.0];
        let se = make_se(3, vals);
        let coords = classical_mds(&se, 2).unwrap();
        let d01 = euclidean_dist(&coords.data, 0, 1, 2);
        let d02 = euclidean_dist(&coords.data, 0, 2, 2);
        assert!(
            d02 > d01,
            "dist(0,2)={d02} should > dist(0,1)={d01} since input d(0,2)=5 > d(0,1)=1"
        );
    }

    #[test]
    fn test_two_node_exact_embedding() {
        // Two nodes at distance 3 → 1D embedding should give exactly distance 3.
        let vals = vec![0.0, 3.0, 3.0, 0.0];
        let se = make_se(2, vals);
        let coords = classical_mds(&se, 1).unwrap();
        let d = euclidean_dist(&coords.data, 0, 1, 1);
        assert!(
            (d - 3.0).abs() < 1e-6,
            "two-node MDS distance should be exact: got {d}"
        );
    }
}
