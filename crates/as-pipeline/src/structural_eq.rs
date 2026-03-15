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

use crate::types::SeMatrix;

/// Compute the n×n structural-equivalence distance matrix.
///
/// Parallelised via rayon when `n > 50`.
pub fn compute_se_matrix(adjacency: &Array2<f64>, labels: Vec<String>) -> SeMatrix {
    let n = adjacency.nrows();
    assert_eq!(n, adjacency.ncols(), "Adjacency matrix must be square");
    assert_eq!(n, labels.len(), "Label count must match matrix size");

    let mut data = vec![0.0f64; n * n];

    if n > 50 {
        // Compute upper triangle in parallel, then mirror.
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .collect();

        let results: Vec<(usize, usize, f64)> = pairs
            .into_par_iter()
            .map(|(i, j)| {
                let d = se_dist(adjacency, n, i, j);
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
                let d = se_dist(adjacency, n, i, j);
                data[i * n + j] = d;
                data[j * n + i] = d;
            }
        }
    }

    SeMatrix::new(labels, data)
}

#[inline]
fn se_dist(adj: &Array2<f64>, n: usize, i: usize, j: usize) -> f64 {
    let mut sum = 0.0f64;
    for k in 0..n {
        let row_diff = adj[[i, k]] - adj[[j, k]];
        let col_diff = adj[[k, i]] - adj[[k, j]];
        sum += row_diff * row_diff + col_diff * col_diff;
    }
    sum.sqrt()
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
        let se = compute_se_matrix(&adj, labels(3));
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
        let se = compute_se_matrix(&adj, labels(4));
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
        let se = compute_se_matrix(&adj, labels(4));
        assert!(
            se.get(2, 3) < 1e-12,
            "Structurally equivalent nodes 2 and 3 should have SE=0, got {}",
            se.get(2, 3)
        );
    }
}
