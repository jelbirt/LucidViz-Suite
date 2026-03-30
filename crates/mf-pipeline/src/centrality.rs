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

use lv_data::CentralityReport;

#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    cost: f64,
    node: usize,
}

impl State {
    fn new(cost: f64, node: usize) -> Self {
        Self { cost, node }
    }
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
    heap.push(State::new(0.0, s));

    while let Some(State { cost, node: v }) = heap.pop() {
        if cost.is_nan() || cost > dist[v] + EPS || settled[v] {
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
                heap.push(State::new(next, w));
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

    // Dijkstra distances (parallelized across source nodes).
    let dist_matrix: Vec<Vec<f64>> = nodes
        .par_iter()
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
                f64::NAN
            } else {
                reachable.iter().sum::<f64>() / reachable.len() as f64
            }
        })
        .collect();

    // NaN distance (disconnected node) also yields closeness=0.0 since NaN > 1e-15 is false.
    let closeness: Vec<f64> = distance
        .iter()
        .map(|&d| if d > 1e-15 { 1.0 / d } else { 0.0 })
        .collect();

    let betweenness = compute_betweenness_pg(pg);

    // Harmonic centrality.
    let harmonic: Vec<f64> = (0..n)
        .map(|i| {
            if n <= 1 {
                return 0.0;
            }
            let sum: f64 = dist_matrix[i]
                .iter()
                .enumerate()
                .filter(|(j, &d)| *j != i && d.is_finite() && d > 0.0)
                .map(|(_, &d)| 1.0 / d)
                .sum();
            sum / (n - 1) as f64
        })
        .collect();

    // Eigenvector centrality via power iteration on adjacency weights.
    let eigenvector = eigenvector_centrality_pg(pg, n, &nodes);

    // PageRank (undirected graph: symmetric adjacency, so PageRank ∝ degree).
    let pagerank = pagerank_pg(pg, n, &nodes, 0.85, 100, 1e-10);

    CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
        harmonic,
        eigenvector,
        pagerank,
    }
}

/// PageRank via power iteration on the petgraph adjacency.
fn pagerank_pg(
    pg: &UnGraph<String, f64>,
    n: usize,
    nodes: &[NodeIndex],
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    let idx_map: std::collections::HashMap<NodeIndex, usize> =
        nodes.iter().enumerate().map(|(i, &ni)| (ni, i)).collect();

    // Out-degree (weighted sum of edge weights) per node.
    let out_degree: Vec<f64> = nodes
        .iter()
        .map(|&ni| pg.edges(ni).map(|e| *e.weight()).sum())
        .collect();

    let mut pr = vec![1.0 / n as f64; n];
    let teleport = (1.0 - alpha) / n as f64;

    for _ in 0..max_iter {
        let mut new_pr = vec![teleport; n];
        for (j, &nj) in nodes.iter().enumerate() {
            if out_degree[j] > 1e-15 {
                let contrib = alpha * pr[j] / out_degree[j];
                for edge in pg.edges(nj) {
                    if let Some(&i) = idx_map.get(&edge.target()) {
                        new_pr[i] += edge.weight() * contrib;
                    }
                }
            } else {
                let contrib = alpha * pr[j] / n as f64;
                for val in new_pr.iter_mut() {
                    *val += contrib;
                }
            }
        }

        let delta: f64 = pr
            .iter()
            .zip(new_pr.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        pr = new_pr;
        if delta < tol {
            break;
        }
    }

    pr
}

/// Eigenvector centrality via power iteration on the petgraph adjacency.
fn eigenvector_centrality_pg(pg: &UnGraph<String, f64>, n: usize, nodes: &[NodeIndex]) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    let idx_map: std::collections::HashMap<NodeIndex, usize> =
        nodes.iter().enumerate().map(|(i, &ni)| (ni, i)).collect();

    let max_iter = 100;
    let tol = 1e-10;
    let mut x = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..max_iter {
        let mut y = vec![0.0f64; n];
        for (i, &ni) in nodes.iter().enumerate() {
            let mut sum = 0.0;
            for edge in pg.edges(ni) {
                if let Some(&j) = idx_map.get(&edge.target()) {
                    sum += edge.weight() * x[j];
                }
            }
            y[i] = sum;
        }

        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return vec![0.0; n];
        }
        for v in &mut y {
            *v /= norm;
        }

        let delta: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        x = y;
        if delta < tol {
            break;
        }
    }

    let max_val = x.iter().cloned().fold(0.0f64, f64::max);
    if max_val > 1e-15 {
        for v in &mut x {
            *v /= max_val;
        }
    }
    x
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
        let report = compute_centrality_full(&graph, &labels(3));

        assert!(report.betweenness[1] > report.betweenness[0]);
        assert!(report.betweenness[1] > report.betweenness[2]);
    }
}
