//! Graph centrality measures for the AlignSpace pipeline.
//!
//! Computes degree, distance (mean shortest path), closeness, and betweenness
//! centrality from a weighted adjacency matrix.
//!
//! Betweenness uses a weighted parallel Brandes algorithm via rayon.

use ndarray::Array2;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::error::AsError;
use crate::types::CentralityMode;
use lv_data::analysis::CentralityReport;

/// Compute centrality measures from an adjacency matrix.
///
/// `CentralityMode::UndirectedLegacy` preserves the original compatibility
/// contract by scanning only the upper triangle of the adjacency matrix.
/// `CentralityMode::Directed` treats every positive `adj[i,j]` as a directed
/// edge from `i` to `j`.
pub fn compute_centrality(
    adj: &Array2<f64>,
    labels: &[String],
    mode: CentralityMode,
) -> Result<CentralityReport, AsError> {
    let n = adj.nrows();
    if n != adj.ncols() {
        return Err(AsError::DimensionMismatch(format!(
            "Centrality adjacency must be square, got {}x{}",
            n,
            adj.ncols()
        )));
    }
    if n != labels.len() {
        return Err(AsError::DimensionMismatch(format!(
            "Centrality label count {} does not match adjacency size {}",
            labels.len(),
            n
        )));
    }

    let weighted_adj = match mode {
        CentralityMode::UndirectedLegacy => {
            weighted_adjacency_undirected(&build_undirected_graph(adj))
        }
        CentralityMode::Directed => weighted_adjacency_directed(adj),
    };
    let degree: Vec<f64> = weighted_adj
        .iter()
        .map(|neighbors| {
            if n > 1 {
                neighbors.len() as f64 / (n - 1) as f64
            } else {
                0.0
            }
        })
        .collect();

    let dist_matrix = shortest_path_matrix(&weighted_adj, n);

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

    let betweenness =
        parallel_brandes_betweenness(&weighted_adj, n, matches!(mode, CentralityMode::Directed));

    Ok(CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
    })
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

fn build_undirected_graph(adj: &Array2<f64>) -> UnGraph<usize, f64> {
    let n = adj.nrows();
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

    graph
}

fn weighted_adjacency_undirected(graph: &UnGraph<usize, f64>) -> Vec<Vec<(usize, f64)>> {
    let nodes: Vec<NodeIndex> = graph.node_indices().collect();
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

fn weighted_adjacency_directed(adj: &Array2<f64>) -> Vec<Vec<(usize, f64)>> {
    let n = adj.nrows();
    (0..n)
        .map(|i| {
            (0..n)
                .filter_map(|j| {
                    let w = adj[[i, j]];
                    if i != j && w > 0.0 {
                        Some((j, w))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect()
}

fn shortest_path_matrix(weighted_adj: &[Vec<(usize, f64)>], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| shortest_paths_from(weighted_adj, i))
        .collect()
}

fn shortest_paths_from(weighted_adj: &[Vec<(usize, f64)>], s: usize) -> Vec<f64> {
    const EPS: f64 = 1e-12;
    let n = weighted_adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::new();

    dist[s] = 0.0;
    heap.push(State { cost: 0.0, node: s });

    while let Some(State { cost, node: v }) = heap.pop() {
        if cost > dist[v] + EPS {
            continue;
        }
        for &(w, weight) in &weighted_adj[v] {
            if weight <= 0.0 {
                continue;
            }
            let next = dist[v] + (1.0 / weight);
            if next + EPS < dist[w] {
                dist[w] = next;
                heap.push(State {
                    cost: next,
                    node: w,
                });
            }
        }
    }

    dist
}

/// Parallel Brandes betweenness centrality for a weighted graph.
fn parallel_brandes_betweenness(
    weighted_adj: &[Vec<(usize, f64)>],
    n: usize,
    directed: bool,
) -> Vec<f64> {
    let partials: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|s| brandes_source(weighted_adj, n, s))
        .collect();

    let mut between = vec![0.0f64; n];
    for partial in &partials {
        for i in 0..n {
            between[i] += partial[i];
        }
    }

    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64
    } else {
        1.0
    };
    if directed {
        between.iter().map(|&b| b / norm).collect()
    } else {
        between.iter().map(|&b| b / norm).collect()
    }
}

/// Brandes algorithm for a single source `s` using weighted shortest paths.
/// Returns a partial delta vector for all nodes.
fn brandes_source(weighted_adj: &[Vec<(usize, f64)>], n: usize, s: usize) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("n{}", i)).collect()
    }

    #[test]
    fn test_degree_complete_graph() {
        let n = 4;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[[i, j]] = 1.0;
                }
            }
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        for &d in &report.degree {
            assert!((d - 1.0).abs() < 1e-10, "degree={}", d);
        }
    }

    #[test]
    fn test_betweenness_path_graph() {
        let n = 5;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
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
        let n = 3;
        let mut adj = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
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

        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");

        assert!(report.betweenness[1] > report.betweenness[0]);
        assert!(report.betweenness[1] > report.betweenness[2]);
    }

    #[test]
    fn test_directed_centrality_respects_edge_direction() {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 2]] = 1.0;

        let directed = compute_centrality(&adj, &labels(3), CentralityMode::Directed)
            .expect("directed centrality should compute");
        let undirected = compute_centrality(&adj, &labels(3), CentralityMode::UndirectedLegacy)
            .expect("undirected centrality should compute");

        assert_eq!(directed.degree, vec![0.5, 0.5, 0.0]);
        assert_ne!(directed.degree, undirected.degree);
        assert!(directed.distance[2] == 0.0);
    }

    #[test]
    fn test_centrality_rejects_dimension_mismatch() {
        let adj = Array2::<f64>::zeros((2, 3));
        let err = compute_centrality(&adj, &labels(2), CentralityMode::UndirectedLegacy)
            .expect_err("non-square adjacency must fail");
        assert!(err.to_string().contains("square"));
    }
}
