//! SMACOF (Scaling by MAjorizing a COmplicated Function) iterative MDS.

use anyhow::Result;
use rayon::prelude::*;

use crate::error::AsError;
use crate::mds::classical::{classical_mds, euclidean_dist, kruskal_stress};
use crate::types::{MdsAlgorithm, MdsCoordinates, SeMatrix, SmacofConfig, SmacofInit};

/// Run SMACOF iterative MDS.
pub fn smacof(dist: &SeMatrix, dims: usize, cfg: &SmacofConfig) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        anyhow::bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.min(n - 1);

    // Initialise X from Classical MDS or seeded random.
    let mut x: Vec<f64> = match cfg.init {
        SmacofInit::Classical => {
            let init_coords = classical_mds(dist, dims)?;
            init_coords.data
        }
        SmacofInit::Random(seed) => random_init(n, dims, seed),
    };

    center_coords(&mut x, n, dims);

    let mut prev_stress = kruskal_stress(dist, &x, n, dims);

    for _iter in 0..cfg.max_iter {
        let new_x = guttman_step(dist, &x, n, dims);
        let cur_stress = kruskal_stress(dist, &new_x, n, dims);

        // Monotone-decrease assertion (allow tiny floating-point slack).
        debug_assert!(
            cur_stress <= prev_stress + 1e-10,
            "SMACOF stress increased: prev={} cur={}",
            prev_stress,
            cur_stress
        );

        let delta = prev_stress - cur_stress;
        x = new_x;
        prev_stress = cur_stress;

        if delta.abs() < cfg.tolerance {
            break;
        }
    }

    center_coords(&mut x, n, dims);
    // Reuse the last computed stress instead of recalculating
    let stress = prev_stress;

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        x,
        dims,
        stress,
        MdsAlgorithm::Smacof,
    )?)
}

/// One Guttman transform step: X_new = (1/N) * B(X) * X
/// Parallelised over rows via rayon.
fn guttman_step(dist: &SeMatrix, x: &[f64], n: usize, dims: usize) -> Vec<f64> {
    let n_f = n as f64;
    // Precompute current pairwise distances (parallelized over rows).
    let d_hat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (0..n)
                .map(|j| euclidean_dist(x, i, j, dims))
                .collect::<Vec<_>>()
        })
        .collect();

    // Compute B matrix, then multiply B*X and scale.
    // B[i,j] = { -delta_ij/d_hat_ij  if i!=j and d_hat>0 | 0 if d_hat=0 | -sum_{k!=i} B[i,k] diagonal }
    // We never store B explicitly; instead each row of B*X is computed on the fly.
    let new_x: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut bx_row = vec![0.0f64; dims];
            let mut diag_sum = 0.0f64;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let delta_ij = dist.get(i, j);
                let dhat_ij = d_hat[i * n + j];
                if dhat_ij > 1e-15 {
                    let b_ij = -delta_ij / dhat_ij;
                    diag_sum -= b_ij; // B[i,i] = -sum_{k!=i} B[i,k]
                    for d in 0..dims {
                        bx_row[d] += b_ij * x[j * dims + d];
                    }
                }
            }
            // Add diagonal contribution.
            for d in 0..dims {
                bx_row[d] += diag_sum * x[i * dims + d];
                bx_row[d] /= n_f;
            }
            bx_row
        })
        .collect();

    new_x
}

fn center_coords(x: &mut [f64], n: usize, dims: usize) {
    for d in 0..dims {
        let mean: f64 = (0..n).map(|i| x[i * dims + d]).sum::<f64>() / n as f64;
        for i in 0..n {
            x[i * dims + d] -= mean;
        }
    }
}

fn random_init(n: usize, dims: usize, seed: u64) -> Vec<f64> {
    // Simple LCG for reproducible pseudo-random initialisation.
    let mut state = seed;
    let mut next = || -> f64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to [-0.5, 0.5]
        ((state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };
    (0..n * dims).map(|_| next()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SmacofInit;

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
    fn test_smacof_stress_monotone() {
        // SMACOF should converge without increasing stress.
        // The debug_assert inside the loop will fire in debug builds if violated.
        let se = line_dist(6);
        let cfg = SmacofConfig {
            max_iter: 50,
            tolerance: 1e-8,
            init: SmacofInit::Classical,
        };
        let coords = smacof(&se, 2, &cfg).unwrap();
        assert!(
            coords.stress < 0.3,
            "stress={} unexpectedly high",
            coords.stress
        );
    }

    #[test]
    fn test_smacof_matches_classical_approx() {
        // For a perfect 1-D layout, SMACOF initialised from Classical should also
        // achieve low stress.
        let se = line_dist(5);
        let cfg = SmacofConfig::default();
        let coords = smacof(&se, 1, &cfg).unwrap();
        assert!(coords.stress < 0.05, "stress={}", coords.stress);
    }
}
