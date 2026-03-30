//! PPMI + Truncated SVD similarity.
//!
//! Applies truncated SVD to the PPMI matrix to produce a denoised, low-rank
//! similarity matrix. The resulting cosine similarities between the low-rank
//! row embeddings replace the raw PPMI values.
//!
//! Reference: Levy, O. & Goldberg, Y. (2014). "Neural Word Embedding as
//! Implicit Matrix Factorization." NeurIPS 27.

use nalgebra::{DMatrix, SVD};

/// Compute a denoised similarity matrix from PPMI via truncated SVD.
///
/// 1. Build the n x n PPMI matrix.
/// 2. Compute full SVD: PPMI = U * Sigma * V^T.
/// 3. Truncate to rank `k` (the top-k singular values/vectors).
/// 4. Form row embeddings: E = U_k * sqrt(Sigma_k).
/// 5. Compute cosine similarity on E: sim(i,j) = dot(E_i, E_j) / (||E_i|| * ||E_j||).
///
/// Returns a symmetric n x n similarity matrix with values in [0, 1].
pub fn ppmi_svd_similarity(ppmi: &[f64], n: usize, rank: usize) -> Vec<f64> {
    if n == 0 || rank == 0 {
        return vec![0.0; n * n];
    }

    let k = rank.min(n);

    // Build nalgebra DMatrix from flat row-major PPMI.
    let mat = DMatrix::from_fn(n, n, |i, j| ppmi[i * n + j]);

    // Full SVD.
    let svd = SVD::new(mat, true, true);
    let u = match svd.u {
        Some(ref u) => u,
        None => return ppmi_fallback(ppmi, n),
    };
    let sigma = &svd.singular_values;

    // Build embeddings: E[i, d] = U[i, d] * sqrt(sigma[d]) for d < k.
    let actual_k = k.min(sigma.len());
    let mut embeddings = vec![0.0f64; n * actual_k];
    for i in 0..n {
        for d in 0..actual_k {
            embeddings[i * actual_k + d] = u[(i, d)] * sigma[d].max(0.0).sqrt();
        }
    }

    // Precompute row norms.
    let norms: Vec<f64> = (0..n)
        .map(|i| {
            let row = &embeddings[i * actual_k..(i + 1) * actual_k];
            row.iter().map(|v| v * v).sum::<f64>().sqrt()
        })
        .collect();

    // Cosine similarity matrix.
    let mut sim = vec![0.0f64; n * n];
    for i in 0..n {
        sim[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let dot: f64 = (0..actual_k)
                .map(|d| embeddings[i * actual_k + d] * embeddings[j * actual_k + d])
                .sum();
            let denom = norms[i] * norms[j];
            let cos = if denom > 1e-15 { dot / denom } else { 0.0 };
            // Clamp to [0, 1] — negative cosine is not meaningful for similarity.
            let val = cos.clamp(0.0, 1.0);
            sim[i * n + j] = val;
            sim[j * n + i] = val;
        }
    }

    sim
}

/// Auto-select a reasonable rank for truncated SVD.
///
/// Heuristic: min(n - 1, max(10, n / 5)), capped at 100.
/// For typical LVS vocab sizes (50-2000), this gives rank 10-100.
pub fn auto_svd_rank(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    if n == 2 {
        return 1;
    }
    let candidate = (n / 5).max(10);
    candidate.min(100).min(n - 1)
}

/// Fallback: normalize raw PPMI to [0, 1] if SVD fails.
fn ppmi_fallback(ppmi: &[f64], n: usize) -> Vec<f64> {
    let max_val = ppmi.iter().cloned().fold(0.0f64, f64::max);
    if max_val < 1e-15 {
        return vec![0.0; n * n];
    }
    ppmi.iter()
        .map(|&v| (v / max_val).clamp(0.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_ppmi_produces_identity_similarity() {
        // Identity-like PPMI (diagonal = 1, off-diag = 0).
        let n = 3;
        let ppmi = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let sim = ppmi_svd_similarity(&ppmi, n, 2);
        assert_eq!(sim.len(), 9);
        // Diagonal must be 1.0.
        for i in 0..n {
            assert!((sim[i * n + i] - 1.0).abs() < 1e-10);
        }
        // Off-diagonal: orthogonal rows → cosine = 0.
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        sim[i * n + j].abs() < 1e-10,
                        "sim[{i},{j}] = {} should be ~0",
                        sim[i * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn similar_rows_produce_high_similarity() {
        let n = 3;
        // Rows 0 and 1 are very similar; row 2 is different.
        #[rustfmt::skip]
        let ppmi = vec![
            1.0, 0.8, 0.1,
            0.8, 1.0, 0.1,
            0.1, 0.1, 1.0,
        ];
        let sim = ppmi_svd_similarity(&ppmi, n, 2);
        // sim(0,1) should be high.
        assert!(sim[0 * n + 1] > 0.5, "sim(0,1)={}", sim[0 * n + 1]);
        // sim(0,2) should be lower.
        assert!(
            sim[0 * n + 1] > sim[0 * n + 2],
            "sim(0,1)={} should > sim(0,2)={}",
            sim[0 * n + 1],
            sim[0 * n + 2]
        );
    }

    #[test]
    fn symmetry_and_range() {
        let n = 4;
        let ppmi: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx / n;
                let j = idx % n;
                if i == j {
                    2.0
                } else {
                    ((i + j) as f64 * 0.3).min(1.5)
                }
            })
            .collect();
        let sim = ppmi_svd_similarity(&ppmi, n, 3);
        for i in 0..n {
            for j in 0..n {
                assert!(sim[i * n + j] >= 0.0, "negative similarity");
                assert!(sim[i * n + j] <= 1.0 + 1e-10, "similarity > 1");
                let diff = (sim[i * n + j] - sim[j * n + i]).abs();
                assert!(diff < 1e-12, "not symmetric at [{i},{j}]");
            }
        }
    }

    #[test]
    fn auto_rank_selection() {
        assert_eq!(auto_svd_rank(0), 0);
        assert_eq!(auto_svd_rank(1), 1);
        assert_eq!(auto_svd_rank(2), 1);
        assert_eq!(auto_svd_rank(3), 2);
        assert_eq!(auto_svd_rank(50), 10);
        assert_eq!(auto_svd_rank(100), 20);
        assert_eq!(auto_svd_rank(500), 100);
        assert_eq!(auto_svd_rank(2000), 100);
    }

    #[test]
    fn empty_input() {
        let sim = ppmi_svd_similarity(&[], 0, 5);
        assert!(sim.is_empty());
    }

    #[test]
    fn zero_ppmi_gives_zero_similarity() {
        let n = 3;
        let ppmi = vec![0.0; n * n];
        let sim = ppmi_svd_similarity(&ppmi, n, 2);
        for &v in &sim {
            assert!(v.abs() < 1e-10 || (v - 1.0).abs() < 1e-10);
        }
    }
}
