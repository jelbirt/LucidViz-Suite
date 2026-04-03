//! t-SNE (t-distributed Stochastic Neighbor Embedding).
//!
//! Implements the standard t-SNE algorithm with:
//! - Pairwise affinity computation with adaptive perplexity
//! - Early exaggeration phase
//! - Gradient descent with momentum
//!
//! Reference: van der Maaten & Hinton (2008).

use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::error::AsError;
use crate::types::{DistanceMatrix, MdsAlgorithm, MdsCoordinates};

/// Configuration for t-SNE.
#[derive(Debug, Clone)]
pub struct TsneConfig {
    /// Perplexity (effective number of neighbors). Typical: 5-50.
    pub perplexity: f64,
    /// Number of gradient descent iterations.
    pub max_iter: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Early exaggeration factor (applied for first `exaggeration_iters`).
    pub exaggeration: f64,
    /// Number of iterations with early exaggeration.
    pub exaggeration_iters: u32,
    /// Momentum for the first `momentum_switch_iter` iterations.
    pub initial_momentum: f64,
    /// Momentum after `momentum_switch_iter`.
    pub final_momentum: f64,
    /// Iteration at which momentum switches from initial to final.
    pub momentum_switch_iter: u32,
}

impl Default for TsneConfig {
    fn default() -> Self {
        Self {
            perplexity: 30.0,
            max_iter: 1000,
            learning_rate: 200.0,
            exaggeration: 4.0,
            exaggeration_iters: 250,
            initial_momentum: 0.5,
            final_momentum: 0.8,
            momentum_switch_iter: 250,
        }
    }
}

/// Run t-SNE on a distance matrix.
#[allow(clippy::needless_range_loop)]
pub fn tsne(dist: &DistanceMatrix, dims: usize, config: &TsneConfig) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.clamp(1, 3);

    // Step 1: Compute pairwise affinities P (with perplexity).
    let p = compute_joint_probabilities(dist, config.perplexity);

    // Step 2: Initialize embedding with small random values.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut y: Vec<f64> = (0..n * dims)
        .map(|_| rng.random::<f64>() * 0.01 - 0.005)
        .collect();
    let mut gains = vec![1.0f64; n * dims];
    let mut y_prev = y.clone();

    // Step 3: Gradient descent.
    for iter in 0..config.max_iter {
        let momentum = if iter < config.momentum_switch_iter {
            config.initial_momentum
        } else {
            config.final_momentum
        };

        // Compute Q distribution (Student-t with 1 df).
        let mut q_num = vec![0.0f64; n * n];
        let mut q_sum = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist_sq = 0.0f64;
                for d in 0..dims {
                    let diff = y[i * dims + d] - y[j * dims + d];
                    dist_sq += diff * diff;
                }
                let val = 1.0 / (1.0 + dist_sq);
                q_num[i * n + j] = val;
                q_num[j * n + i] = val;
                q_sum += 2.0 * val;
            }
        }
        q_sum = q_sum.max(1e-12);

        // Apply early exaggeration to P.
        let p_mult = if iter < config.exaggeration_iters {
            config.exaggeration
        } else {
            1.0
        };

        // Compute gradients.
        let gradients: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut grad = vec![0.0f64; dims];
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let p_ij = p[i * n + j] * p_mult;
                    let q_ij = q_num[i * n + j] / q_sum;
                    let mult = 4.0 * (p_ij - q_ij) * q_num[i * n + j];
                    for d in 0..dims {
                        grad[d] += mult * (y[i * dims + d] - y[j * dims + d]);
                    }
                }
                grad
            })
            .collect();

        // Update with momentum and adaptive gains.
        let y_old = y.clone();
        for i in 0..n {
            for d in 0..dims {
                let idx = i * dims + d;
                let grad = gradients[i][d];
                let vel = y_old[idx] - y_prev[idx];

                // Adaptive gain: increase if gradient and velocity disagree.
                if (grad > 0.0) != (vel > 0.0) {
                    gains[idx] = (gains[idx] + 0.2).min(5.0);
                } else {
                    gains[idx] = (gains[idx] * 0.8).max(0.01);
                }

                y[idx] = y_old[idx] + momentum * vel - config.learning_rate * gains[idx] * grad;
            }
        }
        y_prev = y_old;

        // Re-center.
        for d in 0..dims {
            let mean: f64 = (0..n).map(|i| y[i * dims + d]).sum::<f64>() / n as f64;
            for i in 0..n {
                y[i * dims + d] -= mean;
            }
        }
    }

    let stress = crate::mds::classical::kruskal_stress(dist, &y, n, dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        y,
        dims,
        stress,
        MdsAlgorithm::Tsne,
    )?)
}

/// Compute symmetric joint probabilities P from distances with adaptive perplexity.
#[allow(clippy::needless_range_loop)]
fn compute_joint_probabilities(dist: &DistanceMatrix, perplexity: f64) -> Vec<f64> {
    let n = dist.n;
    let target_entropy = perplexity.ln();

    // Compute conditional probabilities p(j|i) for each i.
    let conditionals: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Binary search for sigma that gives the target perplexity.
            let mut beta = 1.0f64; // beta = 1/(2*sigma^2)
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;

            let mut p_i = vec![0.0f64; n];

            for _trial in 0..50 {
                // Compute p(j|i) = exp(-beta * d_ij^2) / sum_k exp(-beta * d_ik^2)
                let mut sum = 0.0f64;
                for j in 0..n {
                    if j == i {
                        p_i[j] = 0.0;
                        continue;
                    }
                    let d = dist.get(i, j);
                    p_i[j] = (-beta * d * d).exp();
                    sum += p_i[j];
                }
                sum = sum.max(1e-12);
                for j in 0..n {
                    p_i[j] /= sum;
                }

                // Compute entropy: H = -sum p_j * log(p_j)
                let entropy: f64 = p_i
                    .iter()
                    .filter(|&&p| p > 1e-12)
                    .map(|&p| -p * p.ln())
                    .sum();

                let diff = entropy - target_entropy;
                if diff.abs() < 1e-5 {
                    break;
                }
                if diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) / 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + beta_min) / 2.0
                    };
                }
            }

            p_i
        })
        .collect();

    // Symmetrize: P_ij = (p(j|i) + p(i|j)) / (2n)
    let mut p = vec![0.0f64; n * n];
    let scale = 1.0 / (2.0 * n as f64);
    for i in 0..n {
        for j in 0..n {
            p[i * n + j] = (conditionals[i][j] + conditionals[j][i]) * scale;
        }
    }
    // Floor to avoid log(0).
    for v in &mut p {
        *v = v.max(1e-12);
    }

    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{i}")).collect()
    }

    fn line_dist(n: usize) -> DistanceMatrix {
        let mut vals = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vals[i * n + j] = (i as f64 - j as f64).abs();
            }
        }
        DistanceMatrix::new(labels(n), vals).expect("test")
    }

    #[test]
    fn tsne_produces_finite_output() {
        let dist = line_dist(15);
        let config = TsneConfig {
            max_iter: 200,
            perplexity: 5.0,
            ..Default::default()
        };
        let coords = tsne(&dist, 2, &config).unwrap();
        assert_eq!(coords.n, 15);
        assert_eq!(coords.dims, 2);
        for val in &coords.data {
            assert!(val.is_finite(), "coordinate must be finite");
        }
        assert!(coords.stress.is_finite());
        assert_eq!(coords.algorithm, MdsAlgorithm::Tsne);
    }

    #[test]
    fn tsne_preserves_local_neighbors() {
        // In a line graph, node 0 should be closer to node 1 than to node 9.
        let dist = line_dist(10);
        let config = TsneConfig {
            max_iter: 500,
            perplexity: 5.0,
            ..Default::default()
        };
        let coords = tsne(&dist, 2, &config).unwrap();
        let d01 = ((coords.data[0] - coords.data[2]).powi(2)
            + (coords.data[1] - coords.data[3]).powi(2))
        .sqrt();
        let d09 = ((coords.data[0] - coords.data[18]).powi(2)
            + (coords.data[1] - coords.data[19]).powi(2))
        .sqrt();
        assert!(
            d01 < d09,
            "node 0 should be closer to node 1 ({d01}) than node 9 ({d09})"
        );
    }
}
