use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityReport {
    pub labels: Vec<String>,
    pub degree: Vec<f64>,
    pub distance: Vec<f64>,
    pub closeness: Vec<f64>,
    pub betweenness: Vec<f64>,
}
