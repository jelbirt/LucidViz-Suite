//! Centrality computation for the MF pipeline co-occurrence graph.
//!
//! Uses the same weighted shortest-path methodology as AlignSpace,
//! but operates directly on a petgraph `UnGraph<String, f64>`.

use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

    // --- Parallel weighted Brandes betweenness ---
    let betweenness = compute_betweenness_pg(pg);

    CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Full betweenness computation using a weighted adjacency list.
pub(crate) fn betweenness_from_adj_list(adj: &[Vec<(usize, f64)>], n: usize) -> Vec<f64> {
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

fn brandes_source_adj(adj: &[Vec<(usize, f64)>], n: usize, s: usize) -> Vec<f64> {
    const EPS: f64 = 1e-12;
    let mut sigma = vec![0.0f64; n];
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut heap = BinaryHeap::new();
    let mut settled = vec![false; n];

    sigma[s] = 1.0;
    dist[s] = 0.0;
    heap.push(State { cost: 0.0, node: s });

    while let Some(State { cost, node: v }) = heap.pop() {
        if cost > dist[v] + EPS || settled[v] {
            continue;
        }
        settled[v] = true;
        stack.push(v);
        for &(w, weight) in &adj[v] {
            if weight <= 0.0 {
                continue;
            }
            let next = dist[v] + (1.0 / weight);
            if next + EPS < dist[w] {
                dist[w] = next;
                sigma[w] = sigma[v];
                pred[w].clear();
                pred[w].push(v);
                heap.push(State {
                    cost: next,
                    node: w,
                });
            } else if (next - dist[w]).abs() <= EPS {
                sigma[w] += sigma[v];
                pred[w].push(v);
            }
        }
    }

    let mut delta = vec![0.0f64; n];
    let mut partial = vec![0.0f64; n];
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            if sigma[w] > EPS {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
        }
        if w != s {
            partial[w] = delta[w];
        }
    }

    partial
}

/// Build an adjacency list from a petgraph and compute betweenness via Brandes.
pub fn compute_betweenness_pg(pg: &UnGraph<String, f64>) -> Vec<f64> {
    let n = pg.node_count();
    let nodes: Vec<NodeIndex> = pg.node_indices().collect();

    // Build node-index → position map.
    let idx_map: std::collections::HashMap<NodeIndex, usize> =
        nodes.iter().enumerate().map(|(i, &ni)| (ni, i)).collect();

    // Build weighted adjacency list.
    let adj: Vec<Vec<(usize, f64)>> = nodes
        .iter()
        .map(|&ni| {
            pg.edges(ni)
                .filter_map(|edge| {
                    idx_map
                        .get(&edge.target())
                        .copied()
                        .map(|idx| (idx, *edge.weight()))
                })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_from_weighted_edges(n: usize, edges: &[(usize, usize, f64)]) -> UnGraph<String, f64> {
        let mut graph = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|i| graph.add_node(format!("n{i}"))).collect();
        for &(a, b, w) in edges {
            graph.add_edge(nodes[a], nodes[b], w);
        }
        graph
    }

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{i}")).collect()
    }

    #[test]
    fn weighted_betweenness_prefers_stronger_two_hop_path() {
        let graph = graph_from_weighted_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 0.4)]);
        let report = compute_centrality_mf(&graph, &labels(3));

        assert!(report.betweenness[1] > report.betweenness[0]);
        assert!(report.betweenness[1] > report.betweenness[2]);
    }
}
