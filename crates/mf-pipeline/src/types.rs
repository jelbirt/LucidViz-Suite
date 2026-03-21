//! Core types for the MatrixForge pipeline.

use as_pipeline::types::CentralityReport;
pub use lv_data::SimToDistMethod;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single normalised, lowercased, non-stop word token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Token(pub String);

impl Token {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Token {
    fn from(s: &str) -> Self {
        Token(s.to_string())
    }
}

impl From<String> for Token {
    fn from(s: String) -> Self {
        Token(s)
    }
}

// ---------------------------------------------------------------------------
// Co-occurrence graph
// ---------------------------------------------------------------------------

/// Raw token co-occurrence counts in a symmetric vocabulary × vocabulary matrix.
#[derive(Debug, Clone)]
pub struct CooccurrenceGraph {
    pub vocab: Vec<Token>,
    /// Row-major n×n count matrix.
    pub matrix: Vec<u64>,
    pub vocab_size: usize,
    pub window_size: usize,
    pub slide_rate: usize,
}

impl CooccurrenceGraph {
    pub fn new(vocab: Vec<Token>, window_size: usize, slide_rate: usize) -> Self {
        let n = vocab.len();
        CooccurrenceGraph {
            matrix: vec![0u64; n * n],
            vocab_size: n,
            vocab,
            window_size,
            slide_rate,
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> u64 {
        self.matrix[i * self.vocab_size + j]
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: u64) {
        self.matrix[i * self.vocab_size + j] += v;
    }
}

// ---------------------------------------------------------------------------
// MfConfig
// ---------------------------------------------------------------------------

/// Configuration for the MatrixForge pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfConfig {
    /// Context window size (number of tokens on each side).
    pub window_size: usize,
    /// Step size for sliding the window.
    pub slide_rate: usize,
    /// Whether to use PMI weighting.
    pub use_pmi: bool,
    /// Minimum co-occurrence count for a token to enter the vocabulary.
    pub min_count: u64,
    /// Minimum PMI for an edge to exist in the co-occurrence graph.
    pub min_pmi: f64,
    /// BCP 47 language tag for stop-word removal (e.g. `"en"`).
    pub language: String,
    /// Apply Unicode NFC normalization.
    pub unicode_normalize: bool,
    /// Similarity-to-distance conversion method for the AS bridge.
    pub sim_to_dist: SimToDistMethod,
    /// Temporal slicing mode for series output.
    pub slice_mode: MfSliceMode,
    /// Number of tokens per batch when `slice_mode` is `FixedTokenBatch`.
    pub slice_size: usize,
    /// Drop slices smaller than this many post-filtered tokens.
    pub min_tokens_per_slice: usize,
    /// Reuse one global vocabulary across all slices.
    pub shared_vocabulary: bool,
}

impl Default for MfConfig {
    fn default() -> Self {
        MfConfig {
            window_size: 4,
            slide_rate: 1,
            use_pmi: true,
            min_count: 2,
            min_pmi: 0.0,
            language: "en".to_string(),
            unicode_normalize: true,
            sim_to_dist: SimToDistMethod::Linear,
            slice_mode: MfSliceMode::None,
            slice_size: 500,
            min_tokens_per_slice: 1,
            shared_vocabulary: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MfSliceMode {
    #[default]
    None,
    PerFile,
    FixedTokenBatch,
}

// ---------------------------------------------------------------------------
// MfOutput
// ---------------------------------------------------------------------------

/// Output of the MatrixForge pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfOutput {
    /// Vocabulary (sorted alphabetically, after filtering).
    pub labels: Vec<String>,
    /// Runtime-selected similarity matrix (n×n, row-major, values in [0, 1]).
    pub similarity_matrix: Vec<f64>,
    /// Similarity-to-distance conversion selected for downstream AlignSpace use.
    pub sim_to_dist: SimToDistMethod,
    /// NPPMI matrix (n×n, row-major, values in [0, 1]).
    pub nppmi_matrix: Vec<f64>,
    /// Raw co-occurrence count matrix (n×n).
    pub raw_counts: Vec<u64>,
    /// PPMI matrix (n×n, before normalisation).
    pub ppmi_matrix: Vec<f64>,
    pub n: usize,
    pub centrality: CentralityReport,
}

/// One slice in a temporal/comparative MatrixForge series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfSlice {
    pub id: String,
    pub label: String,
    pub order: usize,
    pub source_paths: Vec<PathBuf>,
    pub token_count: usize,
    pub output: MfOutput,
}

/// Ordered MatrixForge series output suitable for downstream AlignSpace use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfSeriesOutput {
    pub labels: Vec<String>,
    pub sim_to_dist: SimToDistMethod,
    pub slices: Vec<MfSlice>,
}

// ---------------------------------------------------------------------------
// MfPipelineConfig
// ---------------------------------------------------------------------------

/// Top-level configuration driving the full MF pipeline run.
#[derive(Debug, Clone)]
pub struct MfPipelineConfig {
    /// Paths to input `.txt` files (or a single directory).
    pub input_paths: Vec<std::path::PathBuf>,
    /// Optional output directory for XLSX/JSON artefacts.
    pub output_dir: Option<std::path::PathBuf>,
    pub mf_config: MfConfig,
    pub write_json: bool,
    pub write_xlsx: bool,
}
