//! Landmark MDS — O(nk) incremental embedding.
//!
//! Based on de Silva & Tenenbaum (2004): selects k landmark points, solves
//! Classical MDS on the k×k landmark distance sub-matrix, then projects
//! remaining points into the landmark coordinate space via distance-based
//! triangulation.
//!
//! This enables incremental embedding: new points can be projected into an
//! existing landmark frame without recomputing MDS from scratch.

use anyhow::{bail, Result};
use nalgebra::{DMatrix, SymmetricEigen};

use crate::error::AsError;
use crate::mds::pivot::farthest_point_pivots;
use crate::types::{DistanceMatrix, MdsAlgorithm, MdsCoordinates};

/// Run Landmark MDS on a full distance matrix.
///
/// # Arguments
/// * `dist`       – Symmetric n×n distance matrix.
/// * `dims`       – Number of output dimensions.
/// * `n_landmarks` – Number of landmark points (k). Clamped to `[dims+1, n]`.
///
/// Returns coordinates for all n points with Kruskal stress.
pub fn landmark_mds(
    dist: &DistanceMatrix,
    dims: usize,
    n_landmarks: usize,
) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.min(n - 1);
    let k = n_landmarks.clamp(dims + 1, n);

    // Step 1: Select landmarks via farthest-point sampling.
    let landmark_indices = farthest_point_pivots(dist, k);

    // Step 2: Build k×k landmark distance matrix.
    let mut dk = vec![0.0f64; k * k];
    for (li, &gi) in landmark_indices.iter().enumerate() {
        for (lj, &gj) in landmark_indices.iter().enumerate() {
            dk[li * k + lj] = dist.get(gi, gj);
        }
    }

    // Step 3: Double-centre the landmark distance matrix.
    let dk_sq: Vec<f64> = dk.iter().map(|&d| d * d).collect();
    let mut row_means = vec![0.0f64; k];
    let mut grand_mean = 0.0f64;
    for i in 0..k {
        for j in 0..k {
            row_means[i] += dk_sq[i * k + j];
        }
        row_means[i] /= k as f64;
        grand_mean += row_means[i];
    }
    grand_mean /= k as f64;

    // B = -0.5 * (D² - row_mean - col_mean + grand_mean)
    let mut b_data = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            b_data[i * k + j] =
                -0.5 * (dk_sq[i * k + j] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    // Step 4: Eigendecompose B to get landmark coordinates.
    let b_mat = DMatrix::from_row_slice(k, k, &b_data);
    let eig = SymmetricEigen::new(b_mat);

    // Sort eigenvalues descending and take top `dims`.
    let mut eigen_pairs: Vec<(usize, f64)> = eig.eigenvalues.iter().copied().enumerate().collect();
    eigen_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let actual_dims = dims.min(eigen_pairs.len());
    let mut landmark_coords = vec![0.0f64; k * actual_dims];
    for d in 0..actual_dims {
        let (col_idx, eigenvalue) = eigen_pairs[d];
        let scale = if eigenvalue > 0.0 {
            eigenvalue.sqrt()
        } else {
            0.0
        };
        for i in 0..k {
            landmark_coords[i * actual_dims + d] = eig.eigenvectors[(i, col_idx)] * scale;
        }
    }

    // Step 5: Project all n points using landmark triangulation.
    // For each non-landmark point p:
    //   x_p = L_k^{-1} * (-0.5) * (d_p^2 - column_means_of_D_k^2)
    // where L_k is the dims × k pseudo-inverse of the landmark coordinates.

    // Build the pseudo-inverse: L_k^+ = (L^T L)^{-1} L^T
    let l_mat = DMatrix::from_row_slice(k, actual_dims, &landmark_coords);
    let ltl = l_mat.transpose() * &l_mat;
    let ltl_inv = match ltl.try_inverse() {
        Some(inv) => inv,
        None => {
            // Fall back to classical MDS if pseudo-inverse fails.
            return crate::mds::classical::classical_mds(dist, dims);
        }
    };
    let pseudo_inv = &ltl_inv * l_mat.transpose(); // dims × k

    // Column means of D_k^2 (these are row_means computed above since D_k is symmetric).
    let col_means_dk_sq = &row_means;

    // Project all n points.
    let mut all_coords = vec![0.0f64; n * actual_dims];

    // Place landmarks directly.
    for (li, &gi) in landmark_indices.iter().enumerate() {
        for d in 0..actual_dims {
            all_coords[gi * actual_dims + d] = landmark_coords[li * actual_dims + d];
        }
    }

    // Project non-landmark points.
    let is_landmark: Vec<bool> = {
        let mut v = vec![false; n];
        for &gi in &landmark_indices {
            v[gi] = true;
        }
        v
    };

    for p in 0..n {
        if is_landmark[p] {
            continue;
        }
        // Compute delta_p = -0.5 * (d(p, landmark_j)^2 - col_mean_j)
        let mut delta = vec![0.0f64; k];
        for (j, &lj) in landmark_indices.iter().enumerate() {
            let d = dist.get(p, lj);
            delta[j] = -0.5 * (d * d - col_means_dk_sq[j]);
        }

        // x_p = pseudo_inv * delta
        for d in 0..actual_dims {
            let mut sum = 0.0f64;
            for j in 0..k {
                sum += pseudo_inv[(d, j)] * delta[j];
            }
            all_coords[p * actual_dims + d] = sum;
        }
    }

    // Step 6: Compute stress.
    let stress = crate::mds::classical::kruskal_stress(dist, &all_coords, n, actual_dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        all_coords,
        actual_dims,
        stress,
        MdsAlgorithm::Landmark,
    )?)
}

/// Project new points into an existing landmark embedding without re-solving MDS.
///
/// # Arguments
/// * `landmark_dist`  – k×k distance matrix of the original landmarks.
/// * `landmark_coords` – Existing k×dims landmark coordinates.
/// * `new_distances`  – k-element vector: distance from the new point to each landmark.
/// * `dims`           – Embedding dimensionality.
///
/// Returns the projected coordinates for the new point.
#[allow(clippy::needless_range_loop)]
pub fn project_point(
    landmark_dist: &DistanceMatrix,
    landmark_coords: &[f64],
    new_distances: &[f64],
    dims: usize,
) -> Result<Vec<f64>> {
    let k = landmark_dist.n;
    if new_distances.len() != k {
        bail!(AsError::DimensionMismatch(format!(
            "expected {} distances to landmarks, got {}",
            k,
            new_distances.len()
        )));
    }
    if landmark_coords.len() != k * dims {
        bail!(AsError::DimensionMismatch(format!(
            "landmark_coords length {} doesn't match {}×{}",
            landmark_coords.len(),
            k,
            dims
        )));
    }

    // Column means of landmark D^2.
    let mut col_means = vec![0.0f64; k];
    for j in 0..k {
        for i in 0..k {
            let d = landmark_dist.get(i, j);
            col_means[j] += d * d;
        }
        col_means[j] /= k as f64;
    }

    // Pseudo-inverse of landmark coords.
    let l_mat = DMatrix::from_row_slice(k, dims, landmark_coords);
    let ltl = l_mat.transpose() * &l_mat;
    let ltl_inv = ltl
        .try_inverse()
        .ok_or_else(|| AsError::InvalidMatrix("landmark coordinate matrix is singular".into()))?;
    let pseudo_inv = &ltl_inv * l_mat.transpose();

    // Delta vector.
    let mut delta = vec![0.0f64; k];
    for j in 0..k {
        delta[j] = -0.5 * (new_distances[j] * new_distances[j] - col_means[j]);
    }

    // Project.
    let mut coords = vec![0.0f64; dims];
    for d in 0..dims {
        for j in 0..k {
            coords[d] += pseudo_inv[(d, j)] * delta[j];
        }
    }

    Ok(coords)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{i}")).collect()
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
    fn landmark_mds_produces_valid_output() {
        let n = 20;
        let dist = line_dist(n);
        let coords = landmark_mds(&dist, 2, 8).unwrap();
        assert_eq!(coords.n, n);
        assert_eq!(coords.dims, 2);
        assert_eq!(coords.data.len(), n * 2);
        assert!(coords.stress.is_finite());
        assert_eq!(coords.algorithm, MdsAlgorithm::Landmark);
    }

    #[test]
    fn landmark_mds_all_landmarks_equals_classical() {
        // When k=n, landmark MDS should approximate classical MDS.
        let n = 10;
        let dist = line_dist(n);
        let lm = landmark_mds(&dist, 2, n).unwrap();
        let cl = crate::mds::classical::classical_mds(&dist, 2).unwrap();
        // Stress should be very close.
        assert!(
            (lm.stress - cl.stress).abs() < 0.05,
            "landmark stress {} vs classical {}",
            lm.stress,
            cl.stress
        );
    }

    #[test]
    fn landmark_mds_coordinates_are_finite() {
        let n = 50;
        let dist = line_dist(n);
        let coords = landmark_mds(&dist, 3, 15).unwrap();
        assert_eq!(coords.n, n);
        assert_eq!(coords.dims, 3);
        assert!(coords.stress.is_finite(), "stress must be finite");
        for val in &coords.data {
            assert!(val.is_finite(), "all coordinates must be finite");
        }
    }

    #[test]
    fn project_point_recovers_approximate_position() {
        let n = 10;
        let dist = line_dist(n);
        let k = 6;
        let dims = 2;
        let coords = landmark_mds(&dist, dims, k).unwrap();

        // Use first k points as landmarks.
        let landmark_indices = farthest_point_pivots(&dist, k);
        let mut lm_dist_data = vec![0.0f64; k * k];
        let mut lm_coords = vec![0.0f64; k * dims];
        for (li, &gi) in landmark_indices.iter().enumerate() {
            for (lj, &gj) in landmark_indices.iter().enumerate() {
                lm_dist_data[li * k + lj] = dist.get(gi, gj);
            }
            for d in 0..dims {
                lm_coords[li * dims + d] = coords.data[gi * dims + d];
            }
        }
        let lm_labels: Vec<String> = landmark_indices.iter().map(|&i| format!("n{i}")).collect();
        let lm_dist = DistanceMatrix::new(lm_labels, lm_dist_data).unwrap();

        // Project point 5 (which may or may not be a landmark).
        let target = 5;
        let new_dists: Vec<f64> = landmark_indices
            .iter()
            .map(|&li| dist.get(target, li))
            .collect();
        let projected = project_point(&lm_dist, &lm_coords, &new_dists, dims).unwrap();
        assert_eq!(projected.len(), dims);
        for &v in &projected {
            assert!(v.is_finite(), "projected coordinate must be finite");
        }
    }
}
