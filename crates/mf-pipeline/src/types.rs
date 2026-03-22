//! Core types for the MatrixForge pipeline.

use anyhow::{bail, Result};
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

impl MfOutput {
    pub fn validate(&self) -> Result<()> {
        if self.labels.len() != self.n {
            bail!(
                "MF output label count {} does not match declared size {}",
                self.labels.len(),
                self.n
            );
        }

        validate_square_len("similarity_matrix", self.similarity_matrix.len(), self.n)?;
        validate_square_len("nppmi_matrix", self.nppmi_matrix.len(), self.n)?;
        validate_square_len("ppmi_matrix", self.ppmi_matrix.len(), self.n)?;
        validate_square_len("raw_counts", self.raw_counts.len(), self.n)?;

        if self.centrality.labels.len() != self.n
            || self.centrality.degree.len() != self.n
            || self.centrality.distance.len() != self.n
            || self.centrality.closeness.len() != self.n
            || self.centrality.betweenness.len() != self.n
        {
            bail!(
                "MF output centrality vectors do not match declared size {}",
                self.n
            );
        }

        validate_similarity_matrix("similarity_matrix", &self.similarity_matrix, self.n)?;
        validate_similarity_matrix("nppmi_matrix", &self.nppmi_matrix, self.n)?;
        validate_finite_matrix("ppmi_matrix", &self.ppmi_matrix, self.n)?;
        validate_finite_vector("centrality.degree", &self.centrality.degree)?;
        validate_finite_vector("centrality.distance", &self.centrality.distance)?;
        validate_finite_vector("centrality.closeness", &self.centrality.closeness)?;
        validate_finite_vector("centrality.betweenness", &self.centrality.betweenness)?;

        Ok(())
    }
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

impl MfSeriesOutput {
    pub fn validate(&self) -> Result<()> {
        if self.slices.is_empty() {
            bail!("MF series output must contain at least one slice");
        }

        for slice in &self.slices {
            slice.output.validate()?;
        }
        Ok(())
    }

    pub fn validate_for_as_input(&self) -> Result<()> {
        self.validate()?;

        let first = &self.slices[0].output;
        if !self.labels.is_empty() && self.labels != first.labels {
            bail!(
                "MF series top-level labels do not match slice '{}' labels",
                self.slices[0].label
            );
        }
        if self.sim_to_dist != first.sim_to_dist {
            bail!(
                "MF series top-level sim_to_dist {:?} does not match slice '{}' sim_to_dist {:?}",
                self.sim_to_dist,
                self.slices[0].label,
                first.sim_to_dist
            );
        }

        for slice in &self.slices[1..] {
            if slice.output.labels != first.labels {
                bail!(
                    "MF series slice '{}' labels do not match the shared AS label ordering",
                    slice.label
                );
            }
            if slice.output.n != first.n {
                bail!(
                    "MF series slice '{}' size {} does not match first slice size {}",
                    slice.label,
                    slice.output.n,
                    first.n
                );
            }
            if slice.output.sim_to_dist != first.sim_to_dist {
                bail!(
                    "MF series slice '{}' sim_to_dist {:?} does not match first slice sim_to_dist {:?}",
                    slice.label,
                    slice.output.sim_to_dist,
                    first.sim_to_dist
                );
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centrality(labels: &[&str]) -> CentralityReport {
        CentralityReport {
            labels: labels.iter().map(|label| (*label).to_string()).collect(),
            degree: vec![0.0; labels.len()],
            distance: vec![0.0; labels.len()],
            closeness: vec![0.0; labels.len()],
            betweenness: vec![0.0; labels.len()],
        }
    }

    fn output(labels: &[&str], sim_to_dist: SimToDistMethod) -> MfOutput {
        let n = labels.len();
        MfOutput {
            labels: labels.iter().map(|label| (*label).to_string()).collect(),
            similarity_matrix: vec![0.0; n * n],
            sim_to_dist,
            nppmi_matrix: vec![0.0; n * n],
            raw_counts: vec![0; n * n],
            ppmi_matrix: vec![0.0; n * n],
            n,
            centrality: centrality(labels),
        }
    }

    fn slice(label: &str, labels: &[&str], sim_to_dist: SimToDistMethod) -> MfSlice {
        MfSlice {
            id: label.to_string(),
            label: label.to_string(),
            order: 0,
            source_paths: Vec::new(),
            token_count: 1,
            output: output(labels, sim_to_dist),
        }
    }

    #[test]
    fn series_validate_for_as_input_accepts_consistent_slices() {
        let series = MfSeriesOutput {
            labels: vec!["alpha".into(), "beta".into()],
            sim_to_dist: SimToDistMethod::Linear,
            slices: vec![
                slice("s1", &["alpha", "beta"], SimToDistMethod::Linear),
                slice("s2", &["alpha", "beta"], SimToDistMethod::Linear),
            ],
        };

        series
            .validate_for_as_input()
            .expect("series should validate");
    }

    #[test]
    fn series_validate_for_as_input_rejects_mismatched_label_order() {
        let series = MfSeriesOutput {
            labels: vec!["alpha".into(), "beta".into()],
            sim_to_dist: SimToDistMethod::Linear,
            slices: vec![
                slice("s1", &["alpha", "beta"], SimToDistMethod::Linear),
                slice("s2", &["beta", "alpha"], SimToDistMethod::Linear),
            ],
        };

        let err = series
            .validate_for_as_input()
            .expect_err("label mismatch should fail");
        assert!(err.to_string().contains("labels do not match"));
    }

    #[test]
    fn series_validate_for_as_input_rejects_mismatched_distance_mode() {
        let series = MfSeriesOutput {
            labels: vec!["alpha".into(), "beta".into()],
            sim_to_dist: SimToDistMethod::Linear,
            slices: vec![
                slice("s1", &["alpha", "beta"], SimToDistMethod::Linear),
                slice("s2", &["alpha", "beta"], SimToDistMethod::Cosine),
            ],
        };

        let err = series
            .validate_for_as_input()
            .expect_err("distance mode mismatch should fail");
        assert!(err.to_string().contains("sim_to_dist"));
    }

    #[test]
    fn mf_output_validate_rejects_nonfinite_similarity() {
        let mut output = output(&["alpha", "beta"], SimToDistMethod::Linear);
        output.similarity_matrix = vec![1.0, f64::NAN, f64::NAN, 1.0];

        let err = output
            .validate()
            .expect_err("nonfinite similarity should fail");
        assert!(err
            .to_string()
            .contains("similarity_matrix[0,1] is not finite"));
    }

    #[test]
    fn mf_output_validate_rejects_out_of_range_similarity() {
        let mut output = output(&["alpha", "beta"], SimToDistMethod::Linear);
        output.similarity_matrix = vec![1.0, 1.2, 1.2, 1.0];

        let err = output
            .validate()
            .expect_err("out-of-range similarity should fail");
        assert!(err.to_string().contains("must be in [0, 1]"));
    }

    #[test]
    fn mf_output_validate_rejects_asymmetric_similarity() {
        let mut output = output(&["alpha", "beta"], SimToDistMethod::Linear);
        output.similarity_matrix = vec![1.0, 0.4, 0.7, 1.0];

        let err = output
            .validate()
            .expect_err("asymmetric similarity should fail");
        assert!(err.to_string().contains("not symmetric"));
    }
}

fn validate_square_len(name: &str, len: usize, n: usize) -> Result<()> {
    let expected = n
        .checked_mul(n)
        .ok_or_else(|| anyhow::anyhow!("MF output size overflow for {} labels", n))?;
    if len != expected {
        bail!("MF output {name} expected {expected} values for {n} labels, got {len}");
    }
    Ok(())
}

fn validate_finite_vector(name: &str, values: &[f64]) -> Result<()> {
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            bail!("MF output {name}[{idx}] is not finite");
        }
    }
    Ok(())
}

fn validate_finite_matrix(name: &str, values: &[f64], n: usize) -> Result<()> {
    for i in 0..n {
        for j in 0..n {
            let value = values[i * n + j];
            if !value.is_finite() {
                bail!("MF output {name}[{i},{j}] is not finite");
            }
        }
    }
    Ok(())
}

fn validate_similarity_matrix(name: &str, values: &[f64], n: usize) -> Result<()> {
    validate_finite_matrix(name, values, n)?;
    for i in 0..n {
        for j in 0..n {
            let value = values[i * n + j];
            if !(0.0..=1.0).contains(&value) {
                bail!("MF output {name}[{i},{j}] must be in [0, 1], got {value}");
            }
            if j > i {
                let other = values[j * n + i];
                if (value - other).abs() > 1e-12 {
                    bail!(
                        "MF output {name} is not symmetric at [{i},{j}] ({value}) and [{j},{i}] ({other})"
                    );
                }
            }
        }
    }
    Ok(())
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
