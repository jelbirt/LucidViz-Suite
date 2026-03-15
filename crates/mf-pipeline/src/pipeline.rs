//! Main MatrixForge pipeline orchestration.

use anyhow::{bail, Result};

use crate::centrality::compute_centrality_full;
use crate::cooccurrence::build_cooccurrence;
use crate::error::MfError;
use crate::graph::build_petgraph;
use crate::ingest::{ingest_directory, ingest_files};
use crate::normalize::normalize_text;
use crate::output::{write_mf_json, write_mf_xlsx};
use crate::pmi::compute_pmi;
use crate::stopwords::filter_stopwords;
use crate::tokenize::tokenize;
use crate::types::{MfOutput, MfPipelineConfig};

/// Run the full MatrixForge pipeline.
///
/// # Sequence
/// 1. Ingest text from `config.input_paths` (files or a single directory)
/// 2. Normalize (NFC, lowercase, strip punct, collapse whitespace)
/// 3. Tokenize (unicode_words, filter numeric/single-char)
/// 4. Filter stop-words
/// 5. Build co-occurrence graph
/// 6. Compute NPPMI → similarity matrix
/// 7. Build petgraph co-occurrence network
/// 8. Compute centrality
/// 9. Write outputs (JSON/XLSX) if requested
/// 10. Return `MfOutput`
pub fn run_mf_pipeline(config: &MfPipelineConfig) -> Result<MfOutput> {
    // 1. Ingest
    let raw_text = if config.input_paths.len() == 1 && config.input_paths[0].is_dir() {
        ingest_directory(&config.input_paths[0])?
    } else {
        ingest_files(&config.input_paths)?
    };

    if raw_text.trim().is_empty() {
        bail!(MfError::EmptyCorpus);
    }

    // 2. Normalize
    let normalized = normalize_text(&raw_text);

    // 3. Tokenize
    let tokens = tokenize(&normalized);

    if tokens.is_empty() {
        bail!(MfError::EmptyCorpus);
    }

    // 4. Stop-words
    let tokens = filter_stopwords(tokens, &config.mf_config.language);

    // 5. Co-occurrence
    let cooccurrence = build_cooccurrence(&tokens, &config.mf_config);

    if cooccurrence.vocab_size == 0 {
        bail!(MfError::EmptyVocabulary);
    }

    let n = cooccurrence.vocab_size;
    let labels: Vec<String> = cooccurrence.vocab.iter().map(|t| t.0.clone()).collect();

    // 6. PMI / similarity
    let similarity_matrix = compute_pmi(&cooccurrence);
    let ppmi_matrix = crate::pmi::compute_ppmi(&cooccurrence);
    let raw_counts = cooccurrence.matrix.clone();

    // 7. Build petgraph
    let pg = build_petgraph(&cooccurrence, &similarity_matrix, &config.mf_config);

    // 8. Centrality
    let centrality = compute_centrality_full(&pg, &labels);

    let output = MfOutput {
        labels,
        similarity_matrix,
        raw_counts,
        ppmi_matrix,
        n,
        centrality,
    };

    // 9. Write outputs
    if let Some(out_dir) = &config.output_dir {
        std::fs::create_dir_all(out_dir)?;

        if config.write_json {
            write_mf_json(&output, &out_dir.join("mf_output.json"))?;
        }

        if config.write_xlsx {
            write_mf_xlsx(&output, &out_dir.join("mf_output.xlsx"), &output.raw_counts)?;
        }
    }

    Ok(output)
}
