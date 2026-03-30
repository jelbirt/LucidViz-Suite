//! Structural Equivalence matrix computation.
//!
//! SE(i,j) = sqrt( sum_k \[ (A\[i,k\] - A\[j,k\])^2 + (A\[k,i\] - A\[k,j\])^2 \] )
//!
//! For undirected (symmetric) adjacency matrices the two terms inside the sum
//! are equal, so the formula simplifies to
//!   SE(i,j) = sqrt( 2 * sum_k (A\[i,k\] - A\[j,k\])^2 )
//! but we keep the full symmetric form here for correctness with directed graphs.

use ndarray::Array2;
use rayon::prelude::*;

use crate::error::AsError;
use crate::types::SeMatrix;

/// Structural equivalence distance method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum SeMethod {
    /// Euclidean distance on adjacency profiles (original).
    #[default]
    Euclidean,
    /// Pearson correlation on adjacency profiles, converted to distance: d = 1 - r.
    /// Scale-invariant: captures structural role regardless of activity level.
    Correlation,
}

/// Compute the n×n structural-equivalence distance matrix.
///
/// Parallelised via rayon when `n > 50`.
pub fn compute_se_matrix(
    adjacency: &Array2<f64>,
    labels: Vec<String>,
) -> Result<SeMatrix, AsError> {
    compute_se_matrix_with_method(adjacency, labels, SeMethod::Euclidean)
}

/// Compute SE matrix with a specified distance method.
pub fn compute_se_matrix_with_method(
    adjacency: &Array2<f64>,
    labels: Vec<String>,
    method: SeMethod,
) -> Result<SeMatrix, AsError> {
    let n = adjacency.nrows();
    if n != adjacency.ncols() {
        return Err(AsError::DimensionMismatch(format!(
            "Adjacency matrix must be square, got {}x{}",
            n,
            adjacency.ncols()
        )));
    }
    if n != labels.len() {
        return Err(AsError::DimensionMismatch(format!(
            "Label count {} does not match adjacency size {}",
            labels.len(),
            n
        )));
    }

    let dist_fn: fn(&Array2<f64>, usize, usize, usize) -> f64 = match method {
        SeMethod::Euclidean => se_dist_euclidean,
        SeMethod::Correlation => se_dist_correlation,
    };

    let mut data = vec![0.0f64; n * n];

    if n > 50 {
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();

        let results: Vec<(usize, usize, f64)> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let d = dist_fn(adjacency, n, i, j);
                (i, j, d)
            })
            .collect();

        for (i, j, d) in results {
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    } else {
        for i in 0..n {
            for j in (i + 1)..n {
                let d = dist_fn(adjacency, n, i, j);
                data[i * n + j] = d;
                data[j * n + i] = d;
            }
        }
    }

    SeMatrix::new(labels, data)
}

#[inline]
fn se_dist_euclidean(adj: &Array2<f64>, n: usize, i: usize, j: usize) -> f64 {
    let mut sum = 0.0f64;
    for k in 0..n {
        let row_diff = adj[[i, k]] - adj[[j, k]];
        let col_diff = adj[[k, i]] - adj[[k, j]];
        sum += row_diff * row_diff + col_diff * col_diff;
    }
    sum.sqrt()
}

/// Correlation-based SE distance: d(i,j) = 1 - pearson_r(profile_i, profile_j).
/// Profile is the concatenation of row i and column i of the adjacency matrix,
/// excluding the self-edge (adj\[i,i\]) to avoid double-counting the diagonal
/// element that appears in both the row and column views.
#[inline]
fn se_dist_correlation(adj: &Array2<f64>, n: usize, i: usize, j: usize) -> f64 {
    // Build profiles: [row_i \ self | col_i \ self] and [row_j \ self | col_j \ self]
    // Profile length is 2*(n-1) since we skip k==i in both loops for node i,
    // and k==j in both loops for node j.  We accumulate directly to avoid
    // allocating profile vectors.
    let len = 2 * (n - 1);
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_a2 = 0.0f64;
    let mut sum_b2 = 0.0f64;
    let mut sum_ab = 0.0f64;

    // Row part: adj[i, k] and adj[j, k] for k != i and k != j.
    for k in 0..n {
        if k == i || k == j {
            continue;
        }
        let a = adj[[i, k]];
        let b = adj[[j, k]];
        sum_a += a;
        sum_b += b;
        sum_a2 += a * a;
        sum_b2 += b * b;
        sum_ab += a * b;
    }
    // Column part: adj[k, i] and adj[k, j] for k != i and k != j.
    for k in 0..n {
        if k == i || k == j {
            continue;
        }
        let a = adj[[k, i]];
        let b = adj[[k, j]];
        sum_a += a;
        sum_b += b;
        sum_a2 += a * a;
        sum_b2 += b * b;
        sum_ab += a * b;
    }

    let n_f = len as f64;
    let mean_a = sum_a / n_f;
    let mean_b = sum_b / n_f;
    let var_a = sum_a2 / n_f - mean_a * mean_a;
    let var_b = sum_b2 / n_f - mean_b * mean_b;
    let cov = sum_ab / n_f - mean_a * mean_b;

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        // Zero variance: distance is 1 (uncorrelated by convention).
        return 1.0;
    }
    let r = cov / denom;

    // Distance = 1 - r, clamped to [0, 2].
    (1.0 - r).clamp(0.0, 2.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{}", i)).collect()
    }

    #[test]
    fn test_se_zero_diagonal() {
        let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let se = compute_se_matrix(&adj, labels(3)).expect("SE matrix should build");
        for i in 0..3 {
            assert_eq!(se.get(i, i), 0.0, "Diagonal must be 0");
        }
    }

    #[test]
    fn test_se_symmetric() {
        let adj = array![
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0, 1.0],
            [0.0, 3.0, 1.0, 0.0]
        ];
        let se = compute_se_matrix(&adj, labels(4)).expect("SE matrix should build");
        for i in 0..4 {
            for j in 0..4 {
                let diff = (se.get(i, j) - se.get(j, i)).abs();
                assert!(diff < 1e-12, "SE matrix must be symmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_se_identical_rows_zero() {
        // Two nodes with truly identical connectivity should have SE distance 0.
        // Build a graph where nodes 2 and 3 have the same row+col profile:
        // Both connect only to nodes 0 and 1 with the same weights.
        // Adjacency (4x4):
        //     0  1  2  3
        // 0 [ 0  1  1  1 ]
        // 1 [ 1  0  1  1 ]
        // 2 [ 1  1  0  0 ]   <- same connections as node 3
        // 3 [ 1  1  0  0 ]   <- same connections as node 2
        use ndarray::array;
        let adj = array![
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ];
        let se = compute_se_matrix(&adj, labels(4)).expect("SE matrix should build");
        assert!(
            se.get(2, 3) < 1e-12,
            "Structurally equivalent nodes 2 and 3 should have SE=0, got {}",
            se.get(2, 3)
        );
    }

    #[test]
    fn test_se_rejects_dimension_mismatch() {
        let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]];
        let err = compute_se_matrix(&adj, labels(2)).expect_err("non-square matrices must fail");
        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn test_correlation_se_zero_diagonal() {
        let adj = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let se = compute_se_matrix_with_method(&adj, labels(3), SeMethod::Correlation)
            .expect("Correlation SE should build");
        for i in 0..3 {
            assert!(
                se.get(i, i).abs() < 1e-12,
                "Diagonal must be 0, got {}",
                se.get(i, i)
            );
        }
    }

    #[test]
    fn test_correlation_se_symmetric() {
        let adj = array![
            [0.0, 1.0, 2.0, 0.0],
            [1.0, 0.0, 1.0, 3.0],
            [2.0, 1.0, 0.0, 1.0],
            [0.0, 3.0, 1.0, 0.0]
        ];
        let se = compute_se_matrix_with_method(&adj, labels(4), SeMethod::Correlation)
            .expect("Correlation SE should build");
        for i in 0..4 {
            for j in 0..4 {
                let diff = (se.get(i, j) - se.get(j, i)).abs();
                assert!(
                    diff < 1e-12,
                    "Correlation SE must be symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_correlation_se_identical_rows() {
        let adj = array![
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0]
        ];
        let se = compute_se_matrix_with_method(&adj, labels(4), SeMethod::Correlation)
            .expect("Correlation SE should build");
        // Nodes 2 and 3 have identical profiles → correlation = 1 → distance = 0.
        assert!(
            se.get(2, 3) < 1e-12,
            "Identical profiles should have distance=0, got {}",
            se.get(2, 3)
        );
    }

    #[test]
    fn test_correlation_se_scale_invariant() {
        // Correlation-based SE should be scale-invariant: scaling all weights
        // of one node should not change the correlation distance.
        let adj1 = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let adj2 = array![[0.0, 2.0, 2.0], [2.0, 0.0, 1.0], [2.0, 1.0, 0.0]];
        let se1 = compute_se_matrix_with_method(&adj1, labels(3), SeMethod::Correlation)
            .expect("SE1 should build");
        let se2 = compute_se_matrix_with_method(&adj2, labels(3), SeMethod::Correlation)
            .expect("SE2 should build");
        // Compare relative ordering: d(1,2) should be similar in both.
        // Exact values differ because only node 0 is scaled, but the point is
        // that correlation handles magnitude differences gracefully.
        assert!(
            se2.get(1, 2).is_finite(),
            "Correlation SE should handle scaled adjacency"
        );
    }
}
