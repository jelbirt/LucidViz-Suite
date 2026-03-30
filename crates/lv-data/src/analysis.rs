use serde::{Deserialize, Serialize};

/// Serialize a `Vec<f64>` writing NaN as JSON `null`.
mod nan_as_null {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize)]
    #[serde(transparent)]
    struct F64OrNull(Option<f64>);

    pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
        let wrapped: Vec<F64OrNull> = v
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    F64OrNull(None)
                } else {
                    F64OrNull(Some(x))
                }
            })
            .collect();
        wrapped.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
        let v: Vec<Option<f64>> = Vec::deserialize(d)?;
        Ok(v.into_iter().map(|o| o.unwrap_or(f64::NAN)).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityReport {
    pub labels: Vec<String>,
    pub degree: Vec<f64>,
    #[serde(with = "nan_as_null")]
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
