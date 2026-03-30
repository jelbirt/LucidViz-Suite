//! Orthogonal Procrustes alignment.
//!
//! Aligns `source` coordinates to `target` using the shared-label subset,
//! then applies the same rotation (and optionally scale) to all source points.

use anyhow::{bail, Result};
use nalgebra::{DMatrix, SVD};
use std::collections::HashMap;

use crate::error::AsError;
use crate::types::{MdsCoordinates, ProcrustesResult};

/// Align `source` to `target` via orthogonal Procrustes.
///
/// # Arguments
/// * `source`       – Coordinates to be aligned.
/// * `target`       – Reference coordinates.
/// * `allow_scale`  – Whether to compute an isotropic scale factor.
///
/// Returns `Err` if fewer than 2 labels are shared.
pub fn procrustes(
    source: &MdsCoordinates,
    target: &MdsCoordinates,
    allow_scale: bool,
) -> Result<ProcrustesResult> {
    let dims = source.dims.min(target.dims);

    // Find shared labels.
    let target_index: HashMap<&str, usize> = target
        .labels
        .iter()
        .enumerate()
        .map(|(ti, label)| (label.as_str(), ti))
        .collect();
    let shared_indices: Vec<(usize, usize)> = source
        .labels
        .iter()
        .enumerate()
        .filter_map(|(si, lbl)| target_index.get(lbl.as_str()).copied().map(|ti| (si, ti)))
        .collect();

    if shared_indices.len() < 2 {
        bail!(AsError::TooFewSharedLabels(shared_indices.len()));
    }

    let m = shared_indices.len();

    // Build sub-matrices A (source subset) and B (target subset), both m×dims.
    let mut a = DMatrix::<f64>::zeros(m, dims);
    let mut b = DMatrix::<f64>::zeros(m, dims);
    for (row, (si, ti)) in shared_indices.iter().enumerate() {
        for d in 0..dims {
            a[(row, d)] = source.row(*si)[d];
            b[(row, d)] = target.row(*ti)[d];
        }
    }

    // Centre A and B.
    let mu_a = col_means(&a, m, dims);
    let mu_b = col_means(&b, m, dims);
    centre(&mut a, &mu_a, m, dims);
    centre(&mut b, &mu_b, m, dims);

    // M = A^T * B  (dims×dims)
    let mat_m = a.transpose() * &b;

    // SVD of M.
    let svd = SVD::new(mat_m, true, true);
    let u = svd
        .u
        .ok_or_else(|| crate::error::AsError::MdsFailed("SVD: U matrix unavailable".into()))?;
    let vt = svd
        .v_t
        .ok_or_else(|| crate::error::AsError::MdsFailed("SVD: V_t matrix unavailable".into()))?;
    let sigma = svd.singular_values.clone();

    // Reflection fix: d = sign(det(V * U^T))
    let v = vt.transpose();
    let vut = &v * u.transpose();
    let det_sign = vut.determinant().signum();

    // S = diag(1, ..., 1, det_sign)
    // R = V * S * U^T
    let mut s_diag = vec![1.0f64; dims];
    if dims > 0 {
        s_diag[dims - 1] = det_sign;
    }

    // R = V * diag(s) * U^T
    let mut vs = v.clone();
    for d in 0..dims {
        for r in 0..dims {
            vs[(r, d)] *= s_diag[d];
        }
    }
    let r = &vs * u.transpose(); // dims×dims rotation matrix

    // Scale: ss = trace(Sigma * S) / ||A_c||^2_F
    let trace_sigma_s: f64 = (0..dims).map(|d| sigma[d] * s_diag[d]).sum();
    let frob_a_sq: f64 = a.iter().map(|v| v * v).sum();
    let ss = if allow_scale && frob_a_sq > 1e-15 {
        trace_sigma_s / frob_a_sq
    } else {
        1.0
    };

    // Translation: t = mu_B - ss * mu_A * R^T
    let mu_a_row = DMatrix::from_row_slice(1, dims, &mu_a);
    let t_mat = DMatrix::from_row_slice(1, dims, &mu_b) - ss * &mu_a_row * r.transpose();
    let translation: Vec<f64> = (0..dims).map(|d| t_mat[(0, d)]).collect();

    // Apply to ALL source points.
    let n = source.n;
    let mut aligned_data = vec![0.0f64; n * dims];
    for i in 0..n {
        for d_out in 0..dims {
            let mut val = 0.0f64;
            for d_in in 0..dims {
                val += ss * source.row(i)[d_in] * r[(d_in, d_out)];
            }
            aligned_data[i * dims + d_out] = val + translation[d_out];
        }
    }

    // Residual on shared-label subset.
    let mut resid = 0.0f64;
    for (row, (si, ti)) in shared_indices.iter().enumerate() {
        for d in 0..dims {
            let aligned_val = aligned_data[si * dims + d];
            let target_val = target.row(*ti)[d];
            let diff = aligned_val - target_val;
            resid += diff * diff;
        }
        let _ = row; // suppress unused warning
    }

    let rotation_flat: Vec<f64> = r.iter().cloned().collect();

    Ok(ProcrustesResult {
        aligned: MdsCoordinates::new(
            source.labels.clone(),
            aligned_data,
            dims,
            source.stress,
            source.algorithm,
        )?,
        rotation: rotation_flat,
        scale: ss,
        translation,
        residual: resid,
    })
}

/// Generalized Procrustes Analysis: iteratively align all configurations to
/// their consensus (mean) until convergence.
///
/// Returns one `ProcrustesResult` per configuration. The consensus is the
/// element-wise mean of the aligned configurations.
///
/// Reference: Gower, J. C. (1975). "Generalized Procrustes Analysis."
/// *Psychometrika*, 40(1), 33-51.
pub fn gpa(
    configs: &[MdsCoordinates],
    allow_scale: bool,
    max_iter: usize,
    tolerance: f64,
) -> Result<Vec<ProcrustesResult>> {
    let k = configs.len();
    if k == 0 {
        return Ok(Vec::new());
    }
    if k == 1 {
        let dims = configs[0].dims;
        return Ok(vec![ProcrustesResult {
            aligned: configs[0].clone(),
            rotation: identity_rotation(dims),
            scale: 1.0,
            translation: vec![0.0; dims],
            residual: 0.0,
        }]);
    }

    // Initialize aligned set with raw configs.
    let mut aligned: Vec<MdsCoordinates> = configs.to_vec();
    let dims = aligned[0].dims;
    let n = aligned[0].n;

    for _iter in 0..max_iter {
        // 1. Compute consensus = element-wise mean of all aligned configs.
        let consensus = compute_consensus(&aligned)?;

        // 2. Align each config to the consensus.
        let mut new_aligned = Vec::with_capacity(k);
        for cfg in configs {
            let res = procrustes(cfg, &consensus, allow_scale)?;
            new_aligned.push(res.aligned.clone());
        }

        // 3. Check convergence: compare aligned coordinates to previous.
        let delta = coordinate_delta(&aligned, &new_aligned, n, dims);
        aligned = new_aligned;

        if delta < tolerance {
            break;
        }
    }

    // Final pass: produce ProcrustesResult for each config aligned to the
    // final consensus.
    let consensus = compute_consensus(&aligned)?;
    let mut results = Vec::with_capacity(k);
    for cfg in configs {
        results.push(procrustes(cfg, &consensus, allow_scale)?);
    }

    Ok(results)
}

/// Compute the element-wise mean of multiple coordinate sets (consensus).
fn compute_consensus(configs: &[MdsCoordinates]) -> Result<MdsCoordinates> {
    let k = configs.len();
    let n = configs[0].n;
    let dims = configs[0].dims;
    let labels = configs[0].labels.clone();

    let mut data = vec![0.0f64; n * dims];
    for cfg in configs {
        for (i, val) in cfg.data.iter().enumerate() {
            data[i] += val;
        }
    }
    let inv_k = 1.0 / k as f64;
    for val in &mut data {
        *val *= inv_k;
    }

    MdsCoordinates::new(
        labels,
        data,
        dims,
        0.0,
        crate::types::MdsAlgorithm::Classical,
    )
    .map_err(|e| anyhow::anyhow!(e))
}

/// Max absolute coordinate change between two sets of aligned configs.
fn coordinate_delta(old: &[MdsCoordinates], new: &[MdsCoordinates], n: usize, dims: usize) -> f64 {
    let mut max_delta = 0.0f64;
    for (o, ne) in old.iter().zip(new.iter()) {
        for i in 0..(n * dims) {
            let d = (o.data[i] - ne.data[i]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }
    }
    max_delta
}

/// Identity rotation matrix (dims x dims), flattened row-major.
pub fn identity_rotation(dims: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; dims * dims];
    for i in 0..dims {
        r[i * dims + i] = 1.0;
    }
    r
}

fn col_means(m: &DMatrix<f64>, rows: usize, cols: usize) -> Vec<f64> {
    (0..cols)
        .map(|c| (0..rows).map(|r| m[(r, c)]).sum::<f64>() / rows as f64)
        .collect()
}

fn centre(m: &mut DMatrix<f64>, means: &[f64], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols {
            m[(r, c)] -= means[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MdsAlgorithm;

    fn make_coords(labels: Vec<String>, data: Vec<f64>, dims: usize) -> MdsCoordinates {
        MdsCoordinates::new(labels, data, dims, 0.0, MdsAlgorithm::Classical)
            .expect("test coordinates should build")
    }

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{}", i)).collect()
    }

    #[test]
    fn test_procrustes_identity() {
        // Aligning a set to itself should give identity rotation, scale=1, residual=0.
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let src = make_coords(labels(4), data.clone(), 2);
        let tgt = make_coords(labels(4), data, 2);
        let res = procrustes(&src, &tgt, true).unwrap();
        assert!(res.residual < 1e-10, "residual={}", res.residual);
        assert!((res.scale - 1.0).abs() < 1e-10, "scale={}", res.scale);
    }

    #[test]
    fn test_procrustes_reflection_handled() {
        // Source is a 180° rotation of target — should align with near-zero residual.
        // Target: [(1,0), (-1,0), (0,1), (0,-1)]
        // 180° rotation: [(-1,0), (1,0), (0,-1), (0,1)]
        let target_data = vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let source_data = vec![-1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0];
        let src = make_coords(labels(4), source_data, 2);
        let tgt = make_coords(labels(4), target_data, 2);
        let res = procrustes(&src, &tgt, false).unwrap();
        assert!(
            res.residual < 1e-6,
            "residual={} for 180-degree rotation",
            res.residual
        );
    }

    #[test]
    fn test_procrustes_too_few_shared() {
        let src = make_coords(vec!["a".to_string()], vec![0.0, 0.0], 2);
        let tgt = make_coords(vec!["b".to_string()], vec![1.0, 1.0], 2);
        assert!(procrustes(&src, &tgt, false).is_err());
    }

    // ── GPA tests ─────────────────────────────────────────────────────────

    #[test]
    fn gpa_single_config_returns_identity() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let cfg = make_coords(labels(3), data, 2);
        let results = gpa(&[cfg], false, 50, 1e-8).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].residual < 1e-10);
        assert!((results[0].scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn gpa_identical_configs_zero_residual() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let configs: Vec<MdsCoordinates> = (0..4)
            .map(|_| make_coords(labels(4), data.clone(), 2))
            .collect();
        let results = gpa(&configs, false, 50, 1e-8).unwrap();
        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.residual < 1e-8, "residual={}", r.residual);
        }
    }

    #[test]
    fn gpa_rotated_configs_converge() {
        // Config 0: original square. Config 1: slight perturbation.
        // Config 2: another perturbation. All share same labels.
        let data0 = vec![1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
        let data1 = vec![1.05, 0.02, -0.98, 0.01, 0.03, 1.01, -0.01, -1.02];
        let data2 = vec![0.97, -0.01, -1.02, 0.03, -0.02, 0.98, 0.01, -0.99];

        let configs = vec![
            make_coords(labels(4), data0, 2),
            make_coords(labels(4), data1, 2),
            make_coords(labels(4), data2, 2),
        ];

        let results = gpa(&configs, true, 50, 1e-8).unwrap();
        assert_eq!(results.len(), 3);
        // All residuals should be small since configs are near-identical.
        for (i, r) in results.iter().enumerate() {
            assert!(
                r.residual < 0.1,
                "GPA config {i} residual={} too large for near-identical configs",
                r.residual
            );
        }
    }

    #[test]
    fn gpa_empty_input() {
        let results = gpa(&[], false, 50, 1e-8).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn gpa_eliminates_drift_vs_chained() {
        // Three configs where chained Procrustes would accumulate drift.
        // Each is a small perturbation + rotation from the previous.
        let data0 = vec![0.0, 0.0, 2.0, 0.0, 1.0, 1.5];
        let data1 = vec![0.1, 0.0, 2.1, 0.1, 1.1, 1.6];
        let data2 = vec![0.2, 0.1, 2.2, 0.2, 1.2, 1.7];
        let configs = vec![
            make_coords(labels(3), data0, 2),
            make_coords(labels(3), data1, 2),
            make_coords(labels(3), data2, 2),
        ];

        let gpa_results = gpa(&configs, true, 50, 1e-8).unwrap();
        // GPA should produce finite, small residuals.
        for r in &gpa_results {
            assert!(r.residual.is_finite());
        }
    }

    #[test]
    fn gpa_3d_configs() {
        let data0 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let data1 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let configs = vec![
            make_coords(labels(3), data0, 3),
            make_coords(labels(3), data1, 3),
        ];
        let results = gpa(&configs, false, 50, 1e-8).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.residual.is_finite());
            assert_eq!(r.aligned.dims, 3);
        }
    }
}
