//! Graph centrality measures for the AlignSpace pipeline.
//!
//! Computes degree, distance (mean shortest path), closeness, and betweenness
//! centrality from a weighted adjacency matrix.
//!
//! Betweenness uses a weighted parallel Brandes algorithm via rayon.

use ndarray::Array2;
use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::types::CentralityReport;

/// Compute centrality measures from an adjacency matrix using the pipeline's
/// current undirected centrality contract.
///
/// This intentionally treats the graph as undirected for backward
/// compatibility: only the upper triangle is scanned, so directed asymmetry is
/// not reflected in the centrality report even though other AS/LV stages may
/// preserve directed edges.
///
/// Edge weight threshold: an edge exists if `adj[i,j] > 0`.
/// Dijkstra uses `1 / weight` as edge cost so stronger ties are "shorter".
pub fn compute_centrality(adj: &Array2<f64>, labels: &[String]) -> CentralityReport {
    let n = adj.nrows();
    assert_eq!(n, adj.ncols());
    assert_eq!(n, labels.len());

    // Build petgraph UnGraph<usize, f64> (node=index, edge=weight).
    let mut graph: UnGraph<usize, f64> = UnGraph::new_undirected();
    let nodes: Vec<NodeIndex> = (0..n).map(|i| graph.add_node(i)).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let w = adj[[i, j]];
            if w > 0.0 {
                graph.add_edge(nodes[i], nodes[j], w);
            }
        }
    }

    // --- Degree centrality (normalised by n-1) ---
    let degree: Vec<f64> = (0..n)
        .map(|i| {
            let k = graph.edges(nodes[i]).count() as f64;
            if n > 1 {
                k / (n - 1) as f64
            } else {
                0.0
            }
        })
        .collect();

    // --- Shortest-path distances via Dijkstra (cost = 1/weight) ---
    // dist_matrix[i][j] = shortest path length; f64::INFINITY if unreachable.
    let dist_matrix: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let costs = dijkstra(&graph, nodes[i], None, |e| {
                let w = *e.weight();
                if w > 0.0 {
                    1.0 / w
                } else {
                    f64::INFINITY
                }
            });
            (0..n)
                .map(|j| costs.get(&nodes[j]).copied().unwrap_or(f64::INFINITY))
                .collect()
        })
        .collect();

    // --- Distance centrality (mean shortest path to reachable nodes) ---
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

    // --- Closeness centrality (1 / distance, 0 if isolated) ---
    let closeness: Vec<f64> = distance
        .iter()
        .map(|&d| if d > 1e-15 { 1.0 / d } else { 0.0 })
        .collect();

    // --- Betweenness centrality (parallel weighted Brandes, undirected) ---
    let weighted_adj = weighted_adjacency(&graph, &nodes);
    let betweenness = parallel_brandes_betweenness(&weighted_adj, n);

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

fn weighted_adjacency(graph: &UnGraph<usize, f64>, nodes: &[NodeIndex]) -> Vec<Vec<(usize, f64)>> {
    nodes
        .iter()
        .map(|&node| {
            graph
                .edges(node)
                .map(|edge| (graph[edge.target()], *edge.weight()))
                .collect()
        })
        .collect()
}

/// Parallel Brandes betweenness centrality for a weighted graph.
/// Normalises by (n-1)(n-2)/2 for undirected graphs.
fn parallel_brandes_betweenness(weighted_adj: &[Vec<(usize, f64)>], n: usize) -> Vec<f64> {
    // Each source node contributes a partial delta vector.
    let partials: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|s| brandes_source(weighted_adj, n, s))
        .collect();

    // Sum partials.
    let mut between = vec![0.0f64; n];
    for partial in &partials {
        for i in 0..n {
            between[i] += partial[i];
        }
    }

    // For undirected graphs, divide by 2 (each path counted twice) and
    // normalise by (n-1)(n-2)/2.
    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64 / 2.0
    } else {
        1.0
    };

    between.iter().map(|&b| b / 2.0 / norm).collect()
}

/// Brandes algorithm for a single source `s` using weighted shortest paths.
/// Returns a partial delta vector for all nodes.
fn brandes_source(weighted_adj: &[Vec<(usize, f64)>], n: usize, s: usize) -> Vec<f64> {
    const EPS: f64 = 1e-12;
    let mut sigma = vec![0.0f64; n]; // num shortest paths from s
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
        for &(w, weight) in &weighted_adj[v] {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{}", i)).collect()
    }

    #[test]
    fn test_degree_complete_graph() {
        // All nodes in a complete graph have equal normalised degree = 1.0.
        let n = 4;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
        }
        let report = compute_centrality(&adj, &labels(n));
        for &d in &report.degree {
            assert!((d - 1.0).abs() < 1e-10, "degree={}", d);
        }
    }

    #[test]
    fn test_betweenness_path_graph() {
        // In a path graph 0-1-2-3-4, node 2 (middle) has highest betweenness.
        let n = 5;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        let report = compute_centrality(&adj, &labels(n));
        let max_idx = report
            .betweenness
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2, "Middle node should have highest betweenness");
    }

    #[test]
    fn test_closeness_positive() {
        // All connected nodes should have positive closeness.
        let n = 3;
        let mut adj = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        let report = compute_centrality(&adj, &labels(n));
        for &c in &report.closeness {
            assert!(c > 0.0, "closeness={}", c);
        }
    }

    #[test]
    fn test_weighted_betweenness_prefers_stronger_two_hop_path() {
        let n = 3;
        let mut adj = Array2::<f64>::zeros((n, n));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[0, 2]] = 0.4;
        adj[[2, 0]] = 0.4;

        let report = compute_centrality(&adj, &labels(n));

        assert!(report.betweenness[1] > report.betweenness[0]);
        assert!(report.betweenness[1] > report.betweenness[2]);
    }
}
