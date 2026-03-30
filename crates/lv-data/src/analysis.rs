use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityReport {
    pub labels: Vec<String>,
    pub degree: Vec<f64>,
    pub distance: Vec<f64>,
    pub closeness: Vec<f64>,
    pub betweenness: Vec<f64>,
    /// Harmonic centrality: H(v) = (1/(n-1)) * sum_{u!=v} 1/d(v,u).
    /// Well-defined on disconnected graphs (unreachable nodes contribute 0).
    #[serde(default)]
    pub harmonic: Vec<f64>,
    /// Eigenvector centrality: dominant eigenvector of the adjacency matrix.
    /// Computed via power iteration.
    #[serde(default)]
    pub eigenvector: Vec<f64>,
    /// PageRank centrality (damping = 0.85). Handles disconnected and directed
    /// graphs naturally via random-walk teleportation.
    #[serde(default)]
    pub pagerank: Vec<f64>,
}
