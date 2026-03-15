//! Error types for the AlignSpace pipeline.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AsError {
    #[error("Too few nodes: {0} (need at least 2)")]
    TooFewNodes(usize),

    #[error("Too few shared labels: {0} (need at least 2)")]
    TooFewSharedLabels(usize),

    #[error("Matrix dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("XLSX write error: {0}")]
    Xlsx(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("MDS computation failed: {0}")]
    MdsFailed(String),
}
