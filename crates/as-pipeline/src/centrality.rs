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

    let betweenness =
        parallel_brandes_betweenness(&weighted_adj, n, matches!(mode, CentralityMode::Directed));

    // Harmonic centrality: H(v) = (1/(n-1)) * sum_{u!=v} 1/d(v,u)
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

    // Eigenvector centrality via power iteration on the adjacency matrix.
    let eigenvector = eigenvector_centrality(adj, n);

    // PageRank (damping = 0.85).
    let pagerank = pagerank_centrality(adj, n, 0.85, 100, 1e-10);

    Ok(CentralityReport {
        labels: labels.to_vec(),
        degree,
        distance,
        closeness,
        betweenness,
        harmonic,
        eigenvector,
        pagerank,
    })
}

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
                .map(|edge| {
                    // For undirected edges, `target()` may return `node`
                    // itself depending on storage order. Pick the other end.
                    let neighbour = if edge.source() == node {
                        edge.target()
                    } else {
                        edge.source()
                    };
                    (graph[neighbour], *edge.weight())
                })
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
    heap.push(State::new(0.0, s));

    while let Some(State { cost, node: v }) = heap.pop() {
        if cost.is_nan() || cost > dist[v] + EPS {
            continue;
        }
        for &(w, weight) in &weighted_adj[v] {
            if weight <= 0.0 {
                continue;
            }
            let next = dist[v] + (1.0 / weight);
            if next + EPS < dist[w] {
                dist[w] = next;
                heap.push(State::new(next, w));
            }
        }
    }

    dist
}

/// PageRank centrality via power iteration.
///
/// Uses column-normalized adjacency with damping factor alpha (typically 0.85).
/// Handles disconnected and directed graphs via random teleportation.
fn pagerank_centrality(
    adj: &Array2<f64>,
    n: usize,
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

    // Column sums for normalization (out-degree weighted).
    let col_sums: Vec<f64> = (0..n).map(|j| (0..n).map(|i| adj[[i, j]]).sum()).collect();

    let mut pr = vec![1.0 / n as f64; n];
    let teleport = (1.0 - alpha) / n as f64;

    for _ in 0..max_iter {
        let mut new_pr = vec![teleport; n];
        for j in 0..n {
            if col_sums[j] > 1e-15 {
                let contrib = alpha * pr[j] / col_sums[j];
                for i in 0..n {
                    new_pr[i] += adj[[i, j]] * contrib;
                }
            } else {
                // Dangling node: distribute evenly.
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

/// Eigenvector centrality via power iteration.
///
/// Returns the dominant eigenvector of the adjacency matrix, normalized so
/// the maximum component is 1.0. For disconnected graphs, nodes in
/// non-dominant components will have zero centrality.
fn eigenvector_centrality(adj: &Array2<f64>, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0];
    }

    let max_iter = 100;
    let tol = 1e-10;

    // Initialize with uniform vector.
    let mut x = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..max_iter {
        // Matrix-vector product: y = A * x
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += adj[[i, j]] * x[j];
            }
            y[i] = sum;
        }

        // Normalize by L2 norm.
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return vec![0.0; n];
        }
        for v in &mut y {
            *v /= norm;
        }

        // Check convergence.
        let delta: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        x = y;
        if delta < tol {
            break;
        }
    }

    // Normalize so max = 1.0.
    let max_val = x.iter().cloned().fold(0.0f64, f64::max);
    if max_val > 1e-15 {
        for v in &mut x {
            *v /= max_val;
        }
    }

    x
}

/// Threshold above which approximate betweenness is used.
const APPROX_BETWEENNESS_THRESHOLD: usize = 1000;
/// Number of pivot sources for approximate betweenness.
const APPROX_BETWEENNESS_PIVOTS: usize = 100;

/// Parallel Brandes betweenness centrality for a weighted graph.
/// For n > 1000, uses approximate betweenness with 100 pivot sources
/// (Brandes & Pich 2007) to avoid O(n^2) scaling.
fn parallel_brandes_betweenness(
    weighted_adj: &[Vec<(usize, f64)>],
    n: usize,
    directed: bool,
) -> Vec<f64> {
    let (sources, scale_factor): (Vec<usize>, f64) = if n > APPROX_BETWEENNESS_THRESHOLD {
        // Uniform random pivot selection (unbiased). Deterministic seed derived
        // from graph structure for reproducibility across identical inputs.
        use rand::{seq::SliceRandom, SeedableRng};
        let k = APPROX_BETWEENNESS_PIVOTS.min(n);
        let seed: u64 = n as u64 ^ weighted_adj.iter().map(|v| v.len() as u64).sum::<u64>();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);
        let factor = n as f64 / k as f64;
        (indices, factor)
    } else {
        ((0..n).collect(), 1.0)
    };

    let partials: Vec<Vec<f64>> = sources
        .into_par_iter()
        .map(|s| brandes_source(weighted_adj, n, s))
        .collect();

    let mut between = vec![0.0f64; n];
    for partial in &partials {
        for i in 0..n {
            between[i] += partial[i];
        }
    }

    // Scale up if using approximate (sampled) betweenness.
    for b in &mut between {
        *b *= scale_factor;
    }

    let norm = if n > 2 {
        ((n - 1) * (n - 2)) as f64
    } else {
        1.0
    };
    if directed {
        between.iter().map(|&b| b / norm).collect()
    } else {
        // Undirected: Brandes over all n sources double-counts each unordered
        // pair {s,t}. Normalization denominator for undirected is (n-1)(n-2)/2.
        // Combined: raw / ((n-1)(n-2)) = raw / norm.
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
    heap.push(State::new(0.0, s));

    while let Some(State { cost, node: v }) = heap.pop() {
        if cost.is_nan() || cost > dist[v] + EPS || settled[v] {
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
        assert!(directed.distance[2].is_nan());
    }

    #[test]
    fn test_single_node_graph_no_edges() {
        let adj = Array2::<f64>::zeros((1, 1));
        let report = compute_centrality(&adj, &labels(1), CentralityMode::UndirectedLegacy)
            .expect("single-node centrality should not panic");
        assert_eq!(report.betweenness.len(), 1);
        assert!(
            report.betweenness[0].abs() < 1e-15,
            "single-node betweenness should be zero, got {}",
            report.betweenness[0]
        );
        assert_eq!(report.degree, vec![0.0]);
        assert!(report.distance[0].is_nan());
        assert_eq!(report.closeness, vec![0.0]);
    }

    #[test]
    fn test_harmonic_centrality_computed() {
        let n = 3;
        let mut adj = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        assert_eq!(report.harmonic.len(), n);
        for &h in &report.harmonic {
            assert!(h > 0.0, "harmonic should be positive for connected graph");
        }
    }

    #[test]
    fn test_eigenvector_centrality_complete_graph() {
        let n = 4;
        let mut adj = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        assert_eq!(report.eigenvector.len(), n);
        // Complete graph: all nodes should have equal eigenvector centrality.
        let max_val = report.eigenvector.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (max_val - 1.0).abs() < 1e-6,
            "max eigenvector should be 1.0"
        );
        for &e in &report.eigenvector {
            assert!(
                (e - 1.0).abs() < 1e-6,
                "all nodes in complete graph should have equal eigenvector centrality"
            );
        }
    }

    #[test]
    fn test_eigenvector_star_graph() {
        // Star graph: center (node 0) connected to all others.
        let n = 5;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 1..n {
            adj[[0, i]] = 1.0;
            adj[[i, 0]] = 1.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        // Center should have highest eigenvector centrality.
        assert!(
            report.eigenvector[0] >= report.eigenvector[1],
            "center should dominate: {} vs {}",
            report.eigenvector[0],
            report.eigenvector[1]
        );
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let n = 4;
        let mut adj = Array2::<f64>::ones((n, n));
        for i in 0..n {
            adj[[i, i]] = 0.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        let sum: f64 = report.pagerank.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "PageRank should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_pagerank_star_center_dominates() {
        let n = 5;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 1..n {
            adj[[0, i]] = 1.0;
            adj[[i, 0]] = 1.0;
        }
        let report = compute_centrality(&adj, &labels(n), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        // Center node should have highest PageRank.
        for i in 1..n {
            assert!(
                report.pagerank[0] > report.pagerank[i],
                "center PR={} should > leaf PR={}",
                report.pagerank[0],
                report.pagerank[i]
            );
        }
    }

    #[test]
    fn test_disconnected_node_distance_is_nan() {
        let mut adj = Array2::<f64>::zeros((3, 3));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        let report = compute_centrality(&adj, &labels(3), CentralityMode::UndirectedLegacy)
            .expect("centrality should compute");
        assert!(report.distance[0].is_finite());
        assert!(report.distance[1].is_finite());
        assert!(report.distance[2].is_nan());
        assert_eq!(report.closeness[2], 0.0);
        assert_eq!(report.harmonic[2], 0.0);
    }

    #[test]
    fn test_centrality_rejects_dimension_mismatch() {
        let adj = Array2::<f64>::zeros((2, 3));
        let err = compute_centrality(&adj, &labels(2), CentralityMode::UndirectedLegacy)
            .expect_err("non-square adjacency must fail");
        assert!(err.to_string().contains("square"));
    }
}
