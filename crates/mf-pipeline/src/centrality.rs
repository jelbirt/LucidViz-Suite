//! Centrality computation for the MF pipeline co-occurrence graph.
//!
//! Re-uses the same parallel Brandes algorithm as the AS pipeline,
//! but operates directly on a petgraph `UnGraph<String, f64>`.

use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use rayon::prelude::*;

use as_pipeline::types::CentralityReport;

/// Compute degree, distance, closeness, and betweenness centrality from a
/// petgraph co-occurrence graph.
pub fn compute_centrality_mf(pg: &UnGraph<String, f64>, labels: &[String]) -> CentralityReport {
    let n = pg.node_count();
    let nodes: Vec<NodeIndex> = pg.node_indices().collect();

    // --- Degree centrality ---
    let degree: Vec<f64> = nodes
        .iter()
        .map(|&ni| {
            let k = pg.edges(ni).count() as f64;
            if n > 1 {
                k / (n - 1) as f64
            } else {
                0.0
            }
        })
        .collect();

    // --- Shortest-path distances (Dijkstra, cost = 1/weight) ---
    let dist_matrix: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&src| {
            let costs = dijkstra(pg, src, None, |e| {
                let w = *e.weight();
                if w > 0.0 {
                    1.0 / w
                } else {
                    f64::INFINITY
                }
            });
            nodes
                .iter()
                .map(|&tgt| costs.get(&tgt).copied().unwrap_or(f64::INFINITY))
                .collect()
        })
        .collect();

    // --- Distance / closeness centrality ---
    let distance: Vec<f64> = (0..n)
        .map(|i| {
            let reachable: Vec<f64> = dist_matrix[i]
                .iter()
                .enumerate()
                .filter(|(j, &d)| *j != i && d.is_finite())
                .map(|(_, &d)| d)
                .collect();
            if reachable.is_empty() {
                0.0
            } else {
                reachable.iter().sum::<f64>() / reachable.len() as f64
            }
        })
        .collect();

    let closeness: Vec<f64> = distance
        .iter()
        .map(|&d| if d > 1e-15 { 1.0 / d } else { 0.0 })
        .collect();

    // --- Parallel Brandes betweenness (via adjacency-list Brandes) ---
    let betweenness = compute_betweenness_pg(pg);

    CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
    }
}

/// Full betweenness computation using an adjacency list (array-of-arrays).
pub(crate) fn betweenness_from_adj_list(adj: &[Vec<usize>], n: usize) -> Vec<f64> {
    let partials: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|s| brandes_source_adj(adj, n, s))
        .collect();

    let mut between = vec![0.0f64; n];
    for partial in &partials {
        for i in 0..n {
            between[i] += partial[i];
        }
    }

    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64 / 2.0
    } else {
        1.0
    };
    between.iter().map(|&b| b / 2.0 / norm).collect()
}

fn brandes_source_adj(adj: &[Vec<usize>], n: usize, s: usize) -> Vec<f64> {
    use std::collections::VecDeque;

    let mut sigma = vec![0.0f64; n];
    let mut dist = vec![-1i64; n];
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    sigma[s] = 1.0;
    dist[s] = 0;
    queue.push_back(s);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        for &w in &adj[v] {
            if dist[w] < 0 {
                queue.push_back(w);
                dist[w] = dist[v] + 1;
            }
            if dist[w] == dist[v] + 1 {
                sigma[w] += sigma[v];
                pred[w].push(v);
            }
        }
    }

    let mut delta = vec![0.0f64; n];
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
        }
    }

    delta
}

/// Build an adjacency list from a petgraph and compute betweenness via Brandes.
pub fn compute_betweenness_pg(pg: &UnGraph<String, f64>) -> Vec<f64> {
    let n = pg.node_count();
    let nodes: Vec<NodeIndex> = pg.node_indices().collect();

    // Build node-index → position map.
    let idx_map: std::collections::HashMap<NodeIndex, usize> =
        nodes.iter().enumerate().map(|(i, &ni)| (ni, i)).collect();

    // Build adjacency list (unweighted BFS).
    let adj: Vec<Vec<usize>> = nodes
        .iter()
        .map(|&ni| {
            pg.neighbors(ni)
                .filter_map(|nb| idx_map.get(&nb).copied())
                .collect()
        })
        .collect();

    betweenness_from_adj_list(&adj, n)
}

/// High-level centrality function that uses the pg graph for all measures.
pub fn compute_centrality_full(pg: &UnGraph<String, f64>, labels: &[String]) -> CentralityReport {
    let n = pg.node_count();
    let nodes: Vec<NodeIndex> = pg.node_indices().collect();

    // Degree.
    let degree: Vec<f64> = nodes
        .iter()
        .map(|&ni| {
            let k = pg.edges(ni).count() as f64;
            if n > 1 {
                k / (n - 1) as f64
            } else {
                0.0
            }
        })
        .collect();

    // Dijkstra distances.
    let dist_matrix: Vec<Vec<f64>> = nodes
        .iter()
        .map(|&src| {
            let costs = dijkstra(pg, src, None, |e| {
                let w = *e.weight();
                if w > 0.0 {
                    1.0 / w
                } else {
                    f64::INFINITY
                }
            });
            nodes
                .iter()
                .map(|&tgt| costs.get(&tgt).copied().unwrap_or(f64::INFINITY))
                .collect()
        })
        .collect();

    let distance: Vec<f64> = (0..n)
        .map(|i| {
            let reachable: Vec<f64> = dist_matrix[i]
                .iter()
                .enumerate()
                .filter(|(j, &d)| *j != i && d.is_finite())
                .map(|(_, &d)| d)
                .collect();
            if reachable.is_empty() {
                0.0
            } else {
                reachable.iter().sum::<f64>() / reachable.len() as f64
            }
        })
        .collect();

    let closeness: Vec<f64> = distance
        .iter()
        .map(|&d| if d > 1e-15 { 1.0 / d } else { 0.0 })
        .collect();

    let betweenness = compute_betweenness_pg(pg);

    CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
    }
}
