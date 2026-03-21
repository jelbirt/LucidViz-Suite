use serde::{Deserialize, Serialize};

/// Similarity-to-distance conversion shared by MatrixForge and AlignSpace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimToDistMethod {
    /// d = 1 - s
    Linear,
    /// d = sqrt(1 - s^2)
    Cosine,
    /// d = -ln(s) for s > 0
    Info,
}
