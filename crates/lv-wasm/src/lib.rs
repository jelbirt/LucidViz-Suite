//! `lv-wasm` — WebAssembly bindings for the Lucid Visualization Suite.
//!
//! Exposes the MF (MatrixForge) and AS (AlignSpace) pipelines as
//! `wasm_bindgen` functions that accept and return JSON strings.
//! This allows a web-based frontend (e.g., three.js, WebGPU) to
//! drive the full text-to-3D pipeline without a native binary.

use wasm_bindgen::prelude::*;

use mf_pipeline::cooccurrence::build_cooccurrence;
use mf_pipeline::normalize::normalize_text;
use mf_pipeline::pmi::{compute_count_similarity, compute_pmi, compute_ppmi};
use mf_pipeline::stopwords::filter_stopwords;
use mf_pipeline::svd_similarity::{auto_svd_rank, ppmi_svd_similarity};
use mf_pipeline::tokenize::tokenize;
use mf_pipeline::types::{MfConfig, MfOutput, SimilarityMethod};

use as_pipeline::pipeline::{mf_output_to_distance_matrix, run_distance_pipeline};
use as_pipeline::types::{
    AsDistancePipelineInput, CentralityMode, MdsConfig, MdsDimMode, NormalizationMode,
    ProcrustesMode,
};

// ---------------------------------------------------------------------------
// MF Pipeline (text -> similarity matrix)
// ---------------------------------------------------------------------------

/// Run the MatrixForge pipeline on a raw text corpus.
///
/// # Arguments
/// * `corpus` — raw text (UTF-8)
/// * `config_json` — JSON-serialized `MfConfig` (optional; pass `"{}"` for defaults)
///
/// # Returns
/// JSON-serialized `MfOutput` containing labels, similarity matrix, centrality, etc.
#[wasm_bindgen]
pub fn run_mf(corpus: &str, config_json: &str) -> Result<String, JsError> {
    let config: MfConfig = if config_json.is_empty() || config_json == "{}" {
        MfConfig::default()
    } else {
        serde_json::from_str(config_json).map_err(|e| JsError::new(&e.to_string()))?
    };

    let output = run_mf_in_memory(corpus, &config).map_err(|e| JsError::new(&format!("{e:#}")))?;

    serde_json::to_string(&output).map_err(|e| JsError::new(&e.to_string()))
}

/// Run MF -> AS pipeline end-to-end: text -> similarity -> distance -> MDS -> 3D coordinates.
///
/// # Arguments
/// * `corpus` — raw text (UTF-8)
/// * `mf_config_json` — JSON-serialized `MfConfig` (optional; pass `"{}"` for defaults)
///
/// # Returns
/// JSON object with `coordinates` (array of {label, x, y, z}), `stress`, and `centrality`.
#[wasm_bindgen]
pub fn run_mf_to_coordinates(corpus: &str, mf_config_json: &str) -> Result<String, JsError> {
    let mf_config: MfConfig = if mf_config_json.is_empty() || mf_config_json == "{}" {
        MfConfig::default()
    } else {
        serde_json::from_str(mf_config_json).map_err(|e| JsError::new(&e.to_string()))?
    };

    let mf_output =
        run_mf_in_memory(corpus, &mf_config).map_err(|e| JsError::new(&format!("{e:#}")))?;

    let dist = mf_output_to_distance_matrix(
        mf_output.labels.clone(),
        &mf_output.similarity_matrix,
        mf_output.n,
        mf_output.sim_to_dist,
    )
    .map_err(|e| JsError::new(&format!("{e:#}")))?;

    let input = AsDistancePipelineInput {
        datasets: vec![("slice-0".to_string(), dist)],
        mds_config: MdsConfig::Auto,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: true,
        normalization_mode: NormalizationMode::Independent,
        target_range: 500.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_distance_pipeline(&input).map_err(|e| JsError::new(&format!("{e:#}")))?;

    // Build a compact JSON response suitable for web consumers.
    let coords = &result.coordinates[0];
    let points: Vec<serde_json::Value> = coords
        .labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let row = coords.row(i);
            serde_json::json!({
                "label": label,
                "x": row.first().copied().unwrap_or(0.0),
                "y": row.get(1).copied().unwrap_or(0.0),
                "z": row.get(2).copied().unwrap_or(0.0),
            })
        })
        .collect();

    let response = serde_json::json!({
        "coordinates": points,
        "stress": coords.stress,
        "algorithm": format!("{:?}", coords.algorithm),
        "node_count": coords.n,
    });

    serde_json::to_string(&response).map_err(|e| JsError::new(&e.to_string()))
}

// ---------------------------------------------------------------------------
// AS Pipeline (distance matrix -> MDS -> coordinates)
// ---------------------------------------------------------------------------

/// Run the AlignSpace pipeline on a precomputed distance matrix.
///
/// # Arguments
/// * `labels_json` — JSON array of label strings, e.g. `["word1","word2","word3"]`
/// * `distances_json` — JSON array of f64 values (row-major, n*n), symmetric with zero diagonal
///
/// # Returns
/// JSON with `coordinates` (array of {label, x, y, z}), `stress`, and `algorithm`.
#[wasm_bindgen]
pub fn run_as(labels_json: &str, distances_json: &str) -> Result<String, JsError> {
    let labels: Vec<String> =
        serde_json::from_str(labels_json).map_err(|e| JsError::new(&e.to_string()))?;
    let data: Vec<f64> =
        serde_json::from_str(distances_json).map_err(|e| JsError::new(&e.to_string()))?;

    let dist = as_pipeline::types::DistanceMatrix::new(labels, data)
        .map_err(|e| JsError::new(&format!("{e:#}")))?;

    let input = AsDistancePipelineInput {
        datasets: vec![("slice-0".to_string(), dist)],
        mds_config: MdsConfig::Auto,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: true,
        normalization_mode: NormalizationMode::Independent,
        target_range: 500.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_distance_pipeline(&input).map_err(|e| JsError::new(&format!("{e:#}")))?;

    let coords = &result.coordinates[0];
    let points: Vec<serde_json::Value> = coords
        .labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let row = coords.row(i);
            serde_json::json!({
                "label": label,
                "x": row.first().copied().unwrap_or(0.0),
                "y": row.get(1).copied().unwrap_or(0.0),
                "z": row.get(2).copied().unwrap_or(0.0),
            })
        })
        .collect();

    let response = serde_json::json!({
        "coordinates": points,
        "stress": coords.stress,
        "algorithm": format!("{:?}", coords.algorithm),
        "node_count": coords.n,
    });

    serde_json::to_string(&response).map_err(|e| JsError::new(&e.to_string()))
}

// ---------------------------------------------------------------------------
// Internal: in-memory MF pipeline (no file I/O)
// ---------------------------------------------------------------------------

fn run_mf_in_memory(corpus: &str, config: &MfConfig) -> anyhow::Result<MfOutput> {
    if corpus.trim().is_empty() {
        anyhow::bail!("corpus is empty");
    }

    let normalized = normalize_text(corpus, config.unicode_normalize);
    let tokens = tokenize(&normalized);
    if tokens.is_empty() {
        anyhow::bail!("no tokens after tokenization");
    }

    let tokens = filter_stopwords(tokens, &config.language);
    let mut cooccurrence = build_cooccurrence(&tokens, config);
    if cooccurrence.vocab_size == 0 {
        anyhow::bail!("empty vocabulary after co-occurrence");
    }

    let n = cooccurrence.vocab_size;
    let labels: Vec<String> = cooccurrence.vocab.iter().map(|t| t.0.clone()).collect();

    let (similarity_matrix, nppmi_matrix, ppmi_matrix) = if config.use_pmi {
        match config.similarity_method {
            SimilarityMethod::PpmiSvd => {
                let nppmi = compute_pmi(&cooccurrence);
                let ppmi = compute_ppmi(&cooccurrence);
                let rank = auto_svd_rank(n);
                let svd_sim = ppmi_svd_similarity(&ppmi, n, rank);
                (svd_sim, nppmi, ppmi)
            }
            SimilarityMethod::Nppmi => {
                let nppmi = compute_pmi(&cooccurrence);
                let ppmi = compute_ppmi(&cooccurrence);
                (nppmi.clone(), nppmi, ppmi)
            }
        }
    } else {
        let nppmi = compute_pmi(&cooccurrence);
        let ppmi = compute_ppmi(&cooccurrence);
        (compute_count_similarity(&cooccurrence), nppmi, ppmi)
    };

    let pg = mf_pipeline::graph::build_petgraph(&cooccurrence, &similarity_matrix, config);
    let raw_counts = std::mem::take(&mut cooccurrence.matrix);
    let centrality = mf_pipeline::centrality::compute_centrality_full(&pg, &labels);

    Ok(MfOutput {
        labels,
        similarity_matrix,
        sim_to_dist: config.sim_to_dist,
        nppmi_matrix,
        raw_counts,
        ppmi_matrix,
        n,
        centrality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TEXT: &str = "\
        Networks connect individuals organizations and ideas in complex relationships. \
        Data analysis reveals patterns hidden within large collections of information. \
        Visualization tools help people understand complex data and abstract concepts. \
        Statistical methods provide rigorous ways to analyze and interpret data. \
        Communities form around shared interests values and geographic proximity. \
        Knowledge spreads through social networks and communication channels. \
        Patterns in data often reflect underlying social and physical processes. \
        Networks exhibit clustering small world properties and scale free distributions. \
        Centrality measures identify influential nodes within complex network structures. \
        Semantic similarity between words can be measured using co occurrence statistics.";

    #[test]
    fn mf_in_memory_returns_valid_output() {
        let config = MfConfig::default();
        let output = run_mf_in_memory(SAMPLE_TEXT, &config).expect("should succeed");
        assert!(!output.labels.is_empty());
        assert_eq!(output.similarity_matrix.len(), output.n * output.n);
        for &v in &output.similarity_matrix {
            assert!(v.is_finite(), "similarity value must be finite");
        }
    }

    #[test]
    fn mf_in_memory_to_coordinates() {
        let config = MfConfig::default();
        let mf_output = run_mf_in_memory(SAMPLE_TEXT, &config).expect("mf should succeed");

        let dist = mf_output_to_distance_matrix(
            mf_output.labels.clone(),
            &mf_output.similarity_matrix,
            mf_output.n,
            mf_output.sim_to_dist,
        )
        .expect("distance matrix should build");

        let input = AsDistancePipelineInput {
            datasets: vec![("slice-0".to_string(), dist)],
            mds_config: MdsConfig::Auto,
            procrustes_mode: ProcrustesMode::None,
            mds_dims: MdsDimMode::Fixed(3),
            normalize: true,
            normalization_mode: NormalizationMode::Independent,
            target_range: 500.0,
            procrustes_scale: false,
            centrality_mode: CentralityMode::UndirectedLegacy,
        };

        let result = run_distance_pipeline(&input).expect("AS pipeline should succeed");
        let coords = &result.coordinates[0];
        assert_eq!(coords.dims, 3);
        assert!(coords.n > 0);
        assert!(coords.stress.is_finite());
        for val in &coords.data {
            assert!(val.is_finite(), "coordinate must be finite");
        }
    }

    #[test]
    fn as_pipeline_from_distance_matrix() {
        let labels = vec!["a".into(), "b".into(), "c".into()];
        let data = vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.5, 2.0, 1.5, 0.0];
        let dist = as_pipeline::types::DistanceMatrix::new(labels, data)
            .expect("distance matrix should build");

        let input = AsDistancePipelineInput {
            datasets: vec![("slice-0".to_string(), dist)],
            mds_config: MdsConfig::Auto,
            procrustes_mode: ProcrustesMode::None,
            mds_dims: MdsDimMode::Fixed(3),
            normalize: true,
            normalization_mode: NormalizationMode::Independent,
            target_range: 500.0,
            procrustes_scale: false,
            centrality_mode: CentralityMode::UndirectedLegacy,
        };

        let result = run_distance_pipeline(&input).expect("AS pipeline should succeed");
        assert_eq!(result.coordinates[0].n, 3);
    }

    #[test]
    fn mf_in_memory_rejects_empty_corpus() {
        let config = MfConfig::default();
        let err = run_mf_in_memory("", &config).expect_err("empty corpus should fail");
        assert!(
            err.to_string().contains("empty"),
            "error should mention empty: {}",
            err
        );
    }
}
