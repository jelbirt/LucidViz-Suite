//! Graph centrality measures for the AlignSpace pipeline.
//!
//! Computes degree, distance (mean shortest path), closeness, and betweenness
//! centrality from a weighted adjacency matrix.
//!
//! Betweenness uses a parallel Brandes algorithm via rayon.

use ndarray::Array2;
use petgraph::algo::dijkstra;
use petgraph::graph::{NodeIndex, UnGraph};
use rayon::prelude::*;

use crate::types::CentralityReport;

/// Compute centrality measures from a symmetric adjacency matrix.
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

    // --- Betweenness centrality (parallel Brandes, undirected) ---
    let betweenness = parallel_brandes_betweenness(&graph, &nodes, n);

    CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
    }
}

/// Parallel Brandes betweenness centrality for an unweighted/binary graph.
/// Normalises by (n-1)(n-2)/2 for undirected graphs.
fn parallel_brandes_betweenness(
    graph: &UnGraph<usize, f64>,
    nodes: &[NodeIndex],
    n: usize,
) -> Vec<f64> {
    // Each source node contributes a partial delta vector.
    let partials: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|s| brandes_source(graph, nodes, n, s))
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

/// Brandes algorithm for a single source `s` (unweighted BFS-based).
/// Returns a partial delta vector for all nodes.
fn brandes_source(
    graph: &UnGraph<usize, f64>,
    nodes: &[NodeIndex],
    n: usize,
    s: usize,
) -> Vec<f64> {
    use std::collections::VecDeque;

    let mut sigma = vec![0.0f64; n]; // num shortest paths from s
    let mut dist = vec![-1i64; n]; // BFS distance
    let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    sigma[s] = 1.0;
    dist[s] = 0;
    queue.push_back(s);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        // Neighbours.
        for nb in graph.neighbors(nodes[v]) {
            let w = nb.index();
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
        if w != s {
            // delta[w] is the contribution of w through s
        }
    }

    delta
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
}
