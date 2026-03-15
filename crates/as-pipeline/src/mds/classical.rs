//! Classical (metric) MDS via eigendecomposition of the double-centred
//! distance matrix.
//!
//! Uses ndarray for the double-centring step and faer for eigendecomposition.

use anyhow::{bail, Result};
use faer::{Mat, Side};
use ndarray::Array2;

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

    // Step 1: D² matrix (ndarray).
    let d2: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| {
        let d = dist.get(i, j);
        d * d
    });

    // Step 2: Double-centering → B matrix.
    // B[i,j] = -0.5 * (D2[i,j] - row_mean[i] - col_mean[j] + grand_mean)
    let row_means: Vec<f64> = (0..n).map(|i| d2.row(i).sum() / n as f64).collect();
    let col_means: Vec<f64> = (0..n).map(|j| d2.column(j).sum() / n as f64).collect();
    let grand_mean: f64 = row_means.iter().sum::<f64>() / n as f64;

    // Convert B to faer::Mat for eigendecomposition.
    let b_faer = Mat::<f64>::from_fn(n, n, |i, j| {
        -0.5 * (d2[[i, j]] - row_means[i] - col_means[j] + grand_mean)
    });

    // Step 3: Eigendecomposition of B (symmetric positive semi-definite).
    // faer returns eigenvalues in ascending order.
    let eig = b_faer
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| anyhow::anyhow!("Eigendecomposition failed: {:?}", e))?;

    // eigenvalues: ascending (eig.S() is DiagRef)
    // eigenvectors: eig.U() is MatRef n×n, column d is eigenvector for eigenvalue d

    // Step 4: Take top `dims` eigenvalues (descending), clamp negatives to 0.
    let mut coords_data = vec![0.0f64; n * dims];
    let s_col = eig.S().column_vector();
    for d in 0..dims {
        let idx = n - 1 - d; // ascending → top is at n-1
        let lam = s_col[idx].max(0.0);
        let scale = lam.sqrt();
        for i in 0..n {
            coords_data[i * dims + d] = eig.U()[(i, idx)] * scale;
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
    ))
}

/// Kruskal stress-1: sqrt( sum_{i<j}(d_hat - d)^2 / sum_{i<j} d^2 )
pub(crate) fn kruskal_stress(dist: &SeMatrix, coords: &[f64], n: usize, dims: usize) -> f64 {
    let mut num = 0.0f64;
    let mut denom = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist.get(i, j);
            let d_hat = euclidean_dist(coords, i, j, dims);
            num += (d_hat - d) * (d_hat - d);
            denom += d * d;
        }
    }
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
        SeMatrix::new(labels, vals)
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
}
