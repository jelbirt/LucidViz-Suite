//! UMAP (Uniform Manifold Approximation and Projection).
//!
//! Simplified implementation based on McInnes et al. (2018):
//! 1. Build k-nearest-neighbor graph from distance matrix
//! 2. Compute fuzzy simplicial set (smooth nearest-neighbor distances)
//! 3. Optimize low-dimensional embedding via SGD with attractive/repulsive forces

use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};

use crate::error::AsError;
use crate::types::{DistanceMatrix, MdsAlgorithm, MdsCoordinates};

/// Configuration for UMAP.
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Number of nearest neighbors for the k-NN graph.
    pub n_neighbors: usize,
    /// Minimum distance in the embedding (controls cluster tightness).
    pub min_dist: f64,
    /// Number of SGD epochs.
    pub n_epochs: u32,
    /// Learning rate.
    pub learning_rate: f64,
    /// Repulsion strength relative to attraction.
    pub repulsion_strength: f64,
    /// Number of negative samples per positive sample.
    pub negative_sample_rate: usize,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            min_dist: 0.1,
            n_epochs: 200,
            learning_rate: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
        }
    }
}

/// Run UMAP on a distance matrix.
pub fn umap(dist: &DistanceMatrix, dims: usize, config: &UmapConfig) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.clamp(1, 3);
    let k = config.n_neighbors.min(n - 1).max(1);

    // Step 1: Build k-NN graph with fuzzy weights.
    let (knn_indices, knn_dists) = build_knn(dist, k);
    let graph = build_fuzzy_graph(n, k, &knn_indices, &knn_dists);

    // Step 2: Compute a/b parameters for the embedding distance function.
    // d_embed(x,y) = 1 / (1 + a * ||x-y||^{2b})
    // Fit a,b so that d_embed ≈ 1 when ||x-y|| <= min_dist, ≈ 0 otherwise.
    let (a, b) = fit_ab(config.min_dist);

    // Step 3: Initialize with spectral-like initialization (small random for simplicity).
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut y: Vec<f64> = (0..n * dims)
        .map(|_| rng.random::<f64>() * 10.0 - 5.0)
        .collect();

    // Collect all edges with weights > threshold.
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (i, row) in graph.iter().enumerate() {
        for &(j, w) in row {
            if j > i && w > 1e-6 {
                edges.push((i, j, w));
            }
        }
    }

    // Step 4: SGD optimization.
    let n_edges = edges.len();
    if n_edges == 0 {
        // No edges — return random positions.
        let stress = crate::mds::classical::kruskal_stress(dist, &y, n, dims);
        return Ok(MdsCoordinates::new(
            dist.labels.clone(),
            y,
            dims,
            stress,
            MdsAlgorithm::Umap,
        )?);
    }

    for epoch in 0..config.n_epochs {
        let alpha = config.learning_rate * (1.0 - epoch as f64 / config.n_epochs as f64);

        for &(i, j, weight) in &edges {
            // Attractive force.
            let mut dist_sq = 0.0f64;
            for d in 0..dims {
                let diff = y[i * dims + d] - y[j * dims + d];
                dist_sq += diff * diff;
            }
            let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));

            for d in 0..dims {
                let diff = y[i * dims + d] - y[j * dims + d];
                let grad = weight * grad_coeff * diff;
                y[i * dims + d] -= alpha * grad.clamp(-4.0, 4.0);
                y[j * dims + d] += alpha * grad.clamp(-4.0, 4.0);
            }

            // Negative sampling: push random non-neighbor pairs apart.
            for _ in 0..config.negative_sample_rate {
                let neg = rng.random_range(0..n);
                if neg == i {
                    continue;
                }
                let mut neg_dist_sq = 0.0f64;
                for d in 0..dims {
                    let diff = y[i * dims + d] - y[neg * dims + d];
                    neg_dist_sq += diff * diff;
                }
                neg_dist_sq = neg_dist_sq.max(1e-6);

                let repulsion = config.repulsion_strength * 2.0 * b
                    / ((0.001 + neg_dist_sq) * (1.0 + a * neg_dist_sq.powf(b)));

                for d in 0..dims {
                    let diff = y[i * dims + d] - y[neg * dims + d];
                    let grad = repulsion * diff;
                    y[i * dims + d] += alpha * grad.clamp(-4.0, 4.0);
                }
            }
        }
    }

    let stress = crate::mds::classical::kruskal_stress(dist, &y, n, dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        y,
        dims,
        stress,
        MdsAlgorithm::Umap,
    )?)
}

/// Build k-NN graph from distance matrix. Returns (indices, distances) per node.
fn build_knn(dist: &DistanceMatrix, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = dist.n;
    let mut indices = Vec::with_capacity(n);
    let mut dists = Vec::with_capacity(n);

    for i in 0..n {
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist.get(i, j)))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);

        let (idx, dst): (Vec<_>, Vec<_>) = neighbors.into_iter().unzip();
        indices.push(idx);
        dists.push(dst);
    }

    (indices, dists)
}

/// Build fuzzy simplicial set from k-NN data.
/// Returns adjacency list with fuzzy membership weights.
fn build_fuzzy_graph(
    n: usize,
    k: usize,
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f64>],
) -> Vec<Vec<(usize, f64)>> {
    let mut graph: Vec<Vec<(usize, f64)>> = vec![vec![]; n];

    for i in 0..n {
        // Smooth nearest-neighbor distance: sigma_i such that
        // sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k)
        let rho = knn_dists[i].first().copied().unwrap_or(0.0);
        let target = (k as f64).ln() / std::f64::consts::LN_2;

        let sigma = find_sigma(&knn_dists[i], rho, target);

        for (idx, &j) in knn_indices[i].iter().enumerate() {
            let d = knn_dists[i][idx];
            let w = if d <= rho {
                1.0
            } else {
                (-(d - rho) / sigma.max(1e-10)).exp()
            };
            graph[i].push((j, w));
        }
    }

    // Symmetrize: w_sym = w_ij + w_ji - w_ij * w_ji
    let mut sym_graph: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for i in 0..n {
        for &(j, w_ij) in &graph[i] {
            let w_ji = graph[j]
                .iter()
                .find(|&&(idx, _)| idx == i)
                .map(|&(_, w)| w)
                .unwrap_or(0.0);
            let w_sym = w_ij + w_ji - w_ij * w_ji;
            if w_sym > 1e-6 {
                sym_graph[i].push((j, w_sym));
            }
        }
    }

    sym_graph
}

/// Binary search for sigma that gives the target sum of affinities.
fn find_sigma(dists: &[f64], rho: f64, target: f64) -> f64 {
    let mut lo = 1e-5f64;
    let mut hi = 1000.0f64;
    let mut sigma = 1.0;

    for _ in 0..64 {
        sigma = (lo + hi) / 2.0;
        let sum: f64 = dists
            .iter()
            .map(|&d| {
                if d <= rho {
                    1.0
                } else {
                    (-(d - rho) / sigma).exp()
                }
            })
            .sum();

        if (sum - target).abs() < 1e-5 {
            break;
        }
        if sum > target {
            hi = sigma;
        } else {
            lo = sigma;
        }
    }

    sigma
}

/// Fit a, b parameters for the embedding curve.
/// Uses the approximation from the UMAP paper.
fn fit_ab(min_dist: f64) -> (f64, f64) {
    // Approximate fit: a ≈ 1.93 * min_dist^{-0.34}, b ≈ 0.79
    // These values work well for min_dist in [0.001, 1.0].
    let b = 0.79;
    let a = if min_dist < 0.001 {
        100.0
    } else {
        1.93 * min_dist.powf(-0.34)
    };
    (a, b)
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
    fn umap_produces_finite_output() {
        let dist = line_dist(15);
        let config = UmapConfig {
            n_epochs: 100,
            n_neighbors: 5,
            ..Default::default()
        };
        let coords = umap(&dist, 2, &config).unwrap();
        assert_eq!(coords.n, 15);
        assert_eq!(coords.dims, 2);
        for val in &coords.data {
            assert!(val.is_finite(), "coordinate must be finite");
        }
        assert!(coords.stress.is_finite());
        assert_eq!(coords.algorithm, MdsAlgorithm::Umap);
    }

    #[test]
    fn umap_3d_output() {
        let dist = line_dist(10);
        let config = UmapConfig {
            n_epochs: 200,
            n_neighbors: 5,
            ..Default::default()
        };
        let coords = umap(&dist, 3, &config).unwrap();
        assert_eq!(coords.dims, 3);
        assert_eq!(coords.data.len(), 10 * 3);
        for val in &coords.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn fit_ab_reasonable() {
        let (a, b) = fit_ab(0.1);
        assert!(a > 0.0 && a < 1000.0, "a={a}");
        assert!(b > 0.0 && b < 2.0, "b={b}");
    }
}
