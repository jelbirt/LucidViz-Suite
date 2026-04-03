//! Fruchterman-Reingold force-directed layout.
//!
//! Supports both undirected and directed graphs. For directed graphs,
//! attractive forces are asymmetric: edge `i→j` pulls `j` toward `i`
//! more strongly than `i` toward `j`, creating spatial "flow" patterns.

use std::sync::mpsc;

use anyhow::{bail, Result};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::AsError;
use crate::types::{DistanceMatrix, MdsAlgorithm, MdsCoordinates};

/// Configuration for force-directed layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceDirectedConfig {
    /// Number of iterations.
    pub max_iter: u32,
    /// Initial temperature (controls max displacement per step).
    pub initial_temp: f64,
    /// Cooling factor per iteration (multiplicative).
    pub cooling: f64,
    /// Asymmetry factor for directed edges: 0.0 = symmetric, 1.0 = fully asymmetric.
    /// At 0.5, the target node receives 75% of the attractive force, the source 25%.
    pub directed_asymmetry: f64,
}

impl Default for ForceDirectedConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            initial_temp: 100.0,
            cooling: 0.95,
            directed_asymmetry: 0.5,
        }
    }
}

/// Run Fruchterman-Reingold force-directed layout.
///
/// # Arguments
/// * `dist`    – Distance/adjacency matrix (smaller values = stronger attraction).
/// * `dims`    – Number of output dimensions (typically 2 or 3).
/// * `config`  – Layout configuration.
/// * `directed` – If true, treat `dist[i][j]` as a directed edge from i to j.
pub fn force_directed_layout(
    dist: &DistanceMatrix,
    dims: usize,
    config: &ForceDirectedConfig,
    directed: bool,
) -> Result<MdsCoordinates> {
    force_directed_layout_with_progress(dist, dims, config, directed, None)
}

/// Run force-directed layout with an optional progress callback.
pub fn force_directed_layout_with_progress(
    dist: &DistanceMatrix,
    dims: usize,
    config: &ForceDirectedConfig,
    directed: bool,
    progress: Option<mpsc::Sender<f32>>,
) -> Result<MdsCoordinates> {
    let n = dist.n;
    if n < 2 {
        bail!(AsError::TooFewNodes(n));
    }
    let dims = dims.clamp(2, 3); // FR works best in 2D/3D

    // Area proportional to number of nodes.
    let area = (n as f64) * 100.0;
    let k = (area / n as f64).sqrt(); // Optimal distance between nodes

    // Initialize with random positions.
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut pos: Vec<f64> = (0..n * dims)
        .map(|_| rng.random::<f64>() * k * 2.0 - k)
        .collect();

    let mut temp = config.initial_temp;

    let total_iters = config.max_iter as f32;
    for _iter in 0..config.max_iter {
        if let Some(ref tx) = progress {
            let _ = tx.send(_iter as f32 / total_iters);
        }
        // Compute displacements.
        let displacements: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut disp = vec![0.0f64; dims];

                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    // Vector from i to j.
                    let mut delta = vec![0.0f64; dims];
                    let mut dist_sq = 0.0f64;
                    for d in 0..dims {
                        delta[d] = pos[j * dims + d] - pos[i * dims + d];
                        dist_sq += delta[d] * delta[d];
                    }
                    let distance = dist_sq.sqrt().max(1e-6);

                    // Repulsive force (all pairs): k² / distance
                    let repulsion = k * k / distance;
                    for d in 0..dims {
                        disp[d] -= (delta[d] / distance) * repulsion;
                    }

                    // Attractive force (connected pairs only).
                    let weight = dist.get(i, j);
                    if weight > 1e-12 {
                        // Convert distance to attraction strength.
                        // Smaller distance values = stronger connection = more attraction.
                        let strength = weight;
                        let attraction = distance * distance / k * strength;

                        if directed {
                            // For edge i→j: target j gets more pull.
                            // Node i (source) gets reduced attraction.
                            let source_factor = 1.0 - config.directed_asymmetry * 0.5;
                            for d in 0..dims {
                                disp[d] += (delta[d] / distance) * attraction * source_factor;
                            }
                        } else {
                            for d in 0..dims {
                                disp[d] += (delta[d] / distance) * attraction;
                            }
                        }
                    }

                    // For directed: if j→i edge exists, i gets extra pull toward j.
                    if directed && i != j {
                        let reverse_weight = dist.get(j, i);
                        if reverse_weight > 1e-12 {
                            let attraction = distance * distance / k * reverse_weight;
                            let target_factor = 1.0 + config.directed_asymmetry * 0.5;
                            for d in 0..dims {
                                disp[d] += (delta[d] / distance) * attraction * target_factor;
                            }
                        }
                    }
                }

                disp
            })
            .collect();

        // Apply displacements with temperature limiting.
        for (i, disp) in displacements.iter().enumerate() {
            let disp_mag = disp.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-6);

            for (d, &dv) in disp.iter().enumerate() {
                let capped = (dv / disp_mag) * disp_mag.min(temp);
                pos[i * dims + d] += capped;
            }
        }

        temp *= config.cooling;
    }

    // Compute stress for quality metric.
    let stress = crate::mds::classical::kruskal_stress(dist, &pos, n, dims);

    Ok(MdsCoordinates::new(
        dist.labels.clone(),
        pos,
        dims,
        stress,
        MdsAlgorithm::ForceDirected,
    )?)
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

    fn path_graph_dist(n: usize) -> DistanceMatrix {
        let mut vals = vec![0.0; n * n];
        for i in 0..(n - 1) {
            vals[i * n + (i + 1)] = 1.0;
            vals[(i + 1) * n + i] = 1.0;
        }
        DistanceMatrix::new(labels(n), vals).expect("test distance matrix")
    }

    #[test]
    fn force_directed_produces_finite_coordinates() {
        let dist = path_graph_dist(10);
        let config = ForceDirectedConfig {
            max_iter: 100,
            ..Default::default()
        };
        let coords = force_directed_layout(&dist, 3, &config, false).unwrap();
        assert_eq!(coords.n, 10);
        assert_eq!(coords.dims, 3);
        for val in &coords.data {
            assert!(val.is_finite(), "coordinate must be finite");
        }
        assert!(coords.stress.is_finite());
    }

    #[test]
    fn force_directed_with_asymmetry_config() {
        // Symmetric graph but with directed asymmetry enabled.
        // The asymmetry parameter affects force distribution, not input data.
        let dist = path_graph_dist(5);
        let config = ForceDirectedConfig {
            max_iter: 200,
            directed_asymmetry: 0.8,
            ..Default::default()
        };
        let coords = force_directed_layout(&dist, 2, &config, true).unwrap();
        assert_eq!(coords.n, 5);
        for val in &coords.data {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn force_directed_rejects_single_node() {
        let dist = DistanceMatrix::new(vec!["a".into()], vec![0.0]).unwrap();
        let err = force_directed_layout(&dist, 2, &ForceDirectedConfig::default(), false);
        assert!(err.is_err());
    }

    #[test]
    fn force_directed_separates_disconnected_components() {
        // Two disconnected pairs: (0,1) and (2,3)
        let n = 4;
        let mut vals = vec![0.0; n * n];
        vals[0 * n + 1] = 1.0;
        vals[1 * n + 0] = 1.0;
        vals[2 * n + 3] = 1.0;
        vals[3 * n + 2] = 1.0;
        let dist = DistanceMatrix::new(labels(n), vals).expect("test");
        let config = ForceDirectedConfig {
            max_iter: 300,
            ..Default::default()
        };
        let coords = force_directed_layout(&dist, 2, &config, false).unwrap();

        // Connected nodes should be closer to each other than to the other pair.
        let dist_01 = ((coords.data[0] - coords.data[2]).powi(2)
            + (coords.data[1] - coords.data[3]).powi(2))
        .sqrt();
        let dist_02 = ((coords.data[0] - coords.data[4]).powi(2)
            + (coords.data[1] - coords.data[5]).powi(2))
        .sqrt();
        assert!(
            dist_01 < dist_02,
            "connected nodes (0,1) dist={dist_01} should be closer than disconnected (0,2) dist={dist_02}"
        );
    }
}
