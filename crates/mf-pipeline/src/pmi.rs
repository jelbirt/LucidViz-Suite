//! PMI → PPMI → NPPMI computation.
//!
//! All matrices are symmetric, n×n, row-major.

use rayon::prelude::*;

use crate::types::CooccurrenceGraph;

const EPSILON: f64 = 1e-10;

/// Compute NPPMI (Normalised Positive PMI) from a co-occurrence matrix.
///
/// Returns a symmetric n×n matrix with values in [0, 1].
///
/// Formula:
///  - p_ab = count\[i,j\] / total
///  - p_a  = row_sum\[i\] / total
///  - p_b  = col_sum\[j\] / total  (= row_sum\[j\] for symmetric matrices)
///  - pmi  = log2(p_ab / (p_a * p_b))  if all > 0, else 0
///  - ppmi = max(pmi, 0)
///  - nppmi = ppmi / max(-log2(p_ab), EPSILON)
pub fn compute_pmi(graph: &CooccurrenceGraph) -> Vec<f64> {
    let n = graph.vocab_size;
    if n == 0 {
        return vec![];
    }

    let mat = &graph.matrix;

    // Total count (each pair is counted in both directions, so divide by 2).
    let total: f64 = mat.iter().map(|&v| v as f64).sum::<f64>() / 2.0;
    if total < EPSILON {
        return vec![0.0; n * n];
    }

    // Row sums (= col sums since symmetric).
    let row_sums: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| mat[i * n + j] as f64).sum::<f64>())
        .collect();

    // Parallelize row-wise: each row computes its upper-triangle slice.
    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row_vals = Vec::new();
            for j in i..n {
                let count_ij = mat[i * n + j] as f64;
                if count_ij < EPSILON {
                    continue;
                }

                // Probabilities use `total` (the true pair count, already
                // halved from the symmetric matrix sum). Row sums count each
                // pair once per direction, so they also divide by `total`.
                let p_ab = count_ij / total;
                let p_a = row_sums[i] / total;
                let p_b = row_sums[j] / total;

                if p_a < EPSILON || p_b < EPSILON {
                    continue;
                }

                let pmi = (p_ab / (p_a * p_b)).log2();
                let ppmi = pmi.max(0.0);

                let neg_log_p_ab = -p_ab.log2();
                let nppmi = if neg_log_p_ab > EPSILON {
                    (ppmi / neg_log_p_ab).min(1.0)
                } else {
                    0.0
                };

                row_vals.push((j, nppmi));
            }
            row_vals
        })
        .collect();

    let mut out = vec![0.0f64; n * n];
    for (i, row_vals) in rows.into_iter().enumerate() {
        for (j, nppmi) in row_vals {
            out[i * n + j] = nppmi;
            out[j * n + i] = nppmi;
        }
    }

    out
}

/// Compute a simple count-based similarity matrix in `[0, 1]`.
pub fn compute_count_similarity(graph: &CooccurrenceGraph) -> Vec<f64> {
    let n = graph.vocab_size;
    if n == 0 {
        return vec![];
    }

    let max_count = graph.matrix.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        return vec![0.0; n * n];
    }

    graph
        .matrix
        .iter()
        .map(|&count| count as f64 / max_count as f64)
        .collect()
}

/// Compute PPMI (unnormalised) from a co-occurrence matrix.
pub fn compute_ppmi(graph: &CooccurrenceGraph) -> Vec<f64> {
    let n = graph.vocab_size;
    if n == 0 {
        return vec![];
    }

    let mat = &graph.matrix;
    let total: f64 = mat.iter().map(|&v| v as f64).sum::<f64>() / 2.0;
    if total < EPSILON {
        return vec![0.0; n * n];
    }

    let row_sums: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| mat[i * n + j] as f64).sum::<f64>())
        .collect();

    let rows: Vec<Vec<(usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row_vals = Vec::new();
            for j in i..n {
                let count_ij = mat[i * n + j] as f64;
                if count_ij < EPSILON {
                    continue;
                }

                let p_ab = count_ij / total;
                let p_a = row_sums[i] / total;
                let p_b = row_sums[j] / total;

                if p_a < EPSILON || p_b < EPSILON {
                    continue;
                }

                let pmi = (p_ab / (p_a * p_b)).log2();
                let ppmi = pmi.max(0.0);

                row_vals.push((j, ppmi));
            }
            row_vals
        })
        .collect();

    let mut out = vec![0.0f64; n * n];
    for (i, row_vals) in rows.into_iter().enumerate() {
        for (j, ppmi) in row_vals {
            out[i * n + j] = ppmi;
            out[j * n + i] = ppmi;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cooccurrence::build_cooccurrence;
    use crate::types::MfConfig;
    use crate::types::Token;

    fn tokens(words: &[&str]) -> Vec<Token> {
        words.iter().map(|w| Token(w.to_string())).collect()
    }

    fn basic_config() -> MfConfig {
        MfConfig {
            window_size: 2,
            slide_rate: 1,
            min_count: 1,
            ..MfConfig::default()
        }
    }

    #[test]
    fn test_pmi_symmetry() {
        let toks = tokens(&["alpha", "beta", "gamma", "alpha", "beta", "gamma"]);
        let graph = build_cooccurrence(&toks, &basic_config());
        let pmi = compute_pmi(&graph);
        let n = graph.vocab_size;
        for i in 0..n {
            for j in 0..n {
                let diff = (pmi[i * n + j] - pmi[j * n + i]).abs();
                assert!(diff < 1e-12, "PMI matrix not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_pmi_range() {
        let toks = tokens(&["alpha", "beta", "gamma", "alpha", "beta", "gamma", "delta"]);
        let graph = build_cooccurrence(&toks, &basic_config());
        let pmi = compute_pmi(&graph);
        for (i, &v) in pmi.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0 + 1e-10,
                "NPPMI value {v} out of [0,1] at index {i}"
            );
        }
    }

    #[test]
    fn test_similarity_range() {
        let toks = tokens(&["the", "cat", "sat", "on", "the", "mat", "the", "cat"]);
        let graph = build_cooccurrence(
            &toks,
            &MfConfig {
                min_count: 1,
                ..MfConfig::default()
            },
        );
        let sim = compute_pmi(&graph);
        for &v in &sim {
            assert!(!v.is_nan(), "NaN in similarity matrix");
            assert!(v >= 0.0, "negative value in similarity matrix");
        }
    }

    #[test]
    fn test_count_similarity_range() {
        let toks = tokens(&["alpha", "beta", "alpha", "gamma"]);
        let graph = build_cooccurrence(&toks, &basic_config());
        let sim = compute_count_similarity(&graph);
        for &v in &sim {
            assert!(
                (0.0..=1.0).contains(&v),
                "count similarity should be normalized"
            );
        }
    }

    #[test]
    fn test_pmi_known_property() {
        // "alpha" and "beta" co-occur more than expected (they cluster together),
        // while "gamma" and "delta" appear in separate contexts. With ≥3 words in
        // the vocabulary, the pair that clusters should have positive NPPMI.
        let toks = tokens(&[
            "alpha", "beta", "alpha", "beta", "gamma", "delta", "gamma", "delta",
        ]);
        let graph = build_cooccurrence(&toks, &basic_config());
        let pmi = compute_pmi(&graph);
        let n = graph.vocab_size;
        let alpha_idx = graph.vocab.iter().position(|t| t.0 == "alpha").unwrap();
        let beta_idx = graph.vocab.iter().position(|t| t.0 == "beta").unwrap();
        assert!(
            pmi[alpha_idx * n + beta_idx] > 0.0,
            "Clustered co-occurring words should have positive NPPMI, got {}",
            pmi[alpha_idx * n + beta_idx]
        );
    }

    #[test]
    fn test_ppmi_nonnegative() {
        // PPMI = max(PMI, 0) → must be non-negative.
        let toks = tokens(&["alpha", "beta", "gamma", "delta", "alpha", "gamma"]);
        let graph = build_cooccurrence(&toks, &basic_config());
        let ppmi = compute_ppmi(&graph);
        for (i, &v) in ppmi.iter().enumerate() {
            assert!(v >= 0.0, "PPMI[{i}] = {v} is negative");
        }
    }
}
