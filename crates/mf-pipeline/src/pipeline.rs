//! Main MatrixForge pipeline orchestration.

use anyhow::{bail, Result};
use as_pipeline::pipeline::mf_output_to_distance_matrix;
use as_pipeline::types::{
    AsDistancePipelineInput, MdsConfig, MdsDimMode, NormalizationMode, ProcrustesMode,
};

use crate::centrality::compute_centrality_full;
use crate::cooccurrence::{build_cooccurrence, build_cooccurrence_with_vocab};
use crate::error::MfError;
use crate::graph::build_petgraph;
use crate::ingest::{discover_text_sources, ingest_inputs, TextSource};
use crate::normalize::normalize_text;
use crate::output::{write_mf_json, write_mf_series_json, write_mf_xlsx};
use crate::pmi::{compute_count_similarity, compute_pmi, compute_ppmi};
use crate::stopwords::filter_stopwords;
use crate::tokenize::tokenize;
use crate::types::{MfOutput, MfPipelineConfig, MfSeriesOutput, MfSlice, MfSliceMode, Token};

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
    let raw_text = ingest_inputs(&config.input_paths)?;

    if raw_text.trim().is_empty() {
        bail!(MfError::EmptyCorpus);
    }

    // 2. Normalize
    let normalized = normalize_text(&raw_text, config.mf_config.unicode_normalize);

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
    let nppmi_matrix = compute_pmi(&cooccurrence);
    let similarity_matrix = if config.mf_config.use_pmi {
        nppmi_matrix.clone()
    } else {
        compute_count_similarity(&cooccurrence)
    };
    let ppmi_matrix = compute_ppmi(&cooccurrence);
    let raw_counts = cooccurrence.matrix.clone();

    // 7. Build petgraph
    let pg = build_petgraph(&cooccurrence, &similarity_matrix, &config.mf_config);

    // 8. Centrality
    let centrality = compute_centrality_full(&pg, &labels);

    let output = MfOutput {
        labels,
        similarity_matrix,
        sim_to_dist: config.mf_config.sim_to_dist,
        nppmi_matrix,
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

/// Run MatrixForge in temporal/comparative mode and emit an ordered slice series.
pub fn run_mf_series_pipeline(config: &MfPipelineConfig) -> Result<MfSeriesOutput> {
    let prepared_slices = prepare_slices(config)?;
    if prepared_slices.is_empty() {
        bail!(MfError::EmptyCorpus);
    }

    let shared_vocab = if config.mf_config.shared_vocabulary {
        build_shared_vocab(&prepared_slices, config.mf_config.min_count)
    } else {
        Vec::new()
    };

    if config.mf_config.shared_vocabulary && shared_vocab.is_empty() {
        bail!(MfError::EmptyVocabulary);
    }

    let labels = if config.mf_config.shared_vocabulary {
        shared_vocab.iter().map(|token| token.0.clone()).collect()
    } else {
        Vec::new()
    };

    let slices: Vec<MfSlice> = prepared_slices
        .into_iter()
        .enumerate()
        .map(|(order, slice)| {
            let output = if config.mf_config.shared_vocabulary {
                build_output_with_shared_vocab(&slice.tokens, &shared_vocab, &config.mf_config)?
            } else {
                build_output_from_tokens(&slice.tokens, &config.mf_config)?
            };

            Ok(MfSlice {
                id: slice.id,
                label: slice.label,
                order,
                source_paths: slice.source_paths,
                token_count: slice.tokens.len(),
                output,
            })
        })
        .collect::<Result<_>>()?;

    let output = MfSeriesOutput {
        labels,
        sim_to_dist: config.mf_config.sim_to_dist,
        slices,
    };

    if let Some(out_dir) = &config.output_dir {
        std::fs::create_dir_all(out_dir)?;
        if config.write_json {
            write_mf_series_json(&output, &out_dir.join("mf_series_output.json"))?;
        }
    }

    Ok(output)
}

/// Convert MatrixForge series output into an AlignSpace multi-step distance input.
#[derive(Debug, Clone)]
pub struct MfSeriesAsInputOptions {
    pub mds_config: MdsConfig,
    pub procrustes_mode: ProcrustesMode,
    pub mds_dims: MdsDimMode,
    pub normalize: bool,
    pub normalization_mode: NormalizationMode,
    pub target_range: f64,
    pub procrustes_scale: bool,
}

/// Convert MatrixForge series output into an AlignSpace multi-step distance input.
pub fn mf_series_output_to_as_input(
    output: &MfSeriesOutput,
    options: MfSeriesAsInputOptions,
) -> Result<AsDistancePipelineInput> {
    output.validate()?;
    let datasets = output
        .slices
        .iter()
        .map(|slice| {
            Ok((
                slice.label.clone(),
                mf_output_to_distance_matrix(
                    slice.output.labels.clone(),
                    &slice.output.similarity_matrix,
                    slice.output.n,
                    slice.output.sim_to_dist,
                )
                .map_err(|e| anyhow::anyhow!(e))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(AsDistancePipelineInput {
        datasets,
        mds_config: options.mds_config,
        procrustes_mode: options.procrustes_mode,
        mds_dims: options.mds_dims,
        normalize: options.normalize,
        normalization_mode: options.normalization_mode,
        target_range: options.target_range,
        procrustes_scale: options.procrustes_scale,
    })
}

#[derive(Debug, Clone)]
struct PreparedSlice {
    id: String,
    label: String,
    source_paths: Vec<std::path::PathBuf>,
    tokens: Vec<Token>,
}

fn build_output_from_tokens(
    tokens: &[Token],
    mf_config: &crate::types::MfConfig,
) -> Result<MfOutput> {
    let cooccurrence = build_cooccurrence(tokens, mf_config);
    build_output_from_cooccurrence(cooccurrence, mf_config)
}

fn build_output_with_shared_vocab(
    tokens: &[Token],
    shared_vocab: &[Token],
    mf_config: &crate::types::MfConfig,
) -> Result<MfOutput> {
    let cooccurrence = build_cooccurrence_with_vocab(tokens, shared_vocab, mf_config);
    build_output_from_cooccurrence(cooccurrence, mf_config)
}

fn build_output_from_cooccurrence(
    cooccurrence: crate::types::CooccurrenceGraph,
    mf_config: &crate::types::MfConfig,
) -> Result<MfOutput> {
    if cooccurrence.vocab_size == 0 {
        bail!(MfError::EmptyVocabulary);
    }

    let n = cooccurrence.vocab_size;
    let labels: Vec<String> = cooccurrence.vocab.iter().map(|t| t.0.clone()).collect();
    let nppmi_matrix = compute_pmi(&cooccurrence);
    let similarity_matrix = if mf_config.use_pmi {
        nppmi_matrix.clone()
    } else {
        compute_count_similarity(&cooccurrence)
    };
    let ppmi_matrix = compute_ppmi(&cooccurrence);
    let raw_counts = cooccurrence.matrix.clone();
    let pg = build_petgraph(&cooccurrence, &similarity_matrix, mf_config);
    let centrality = compute_centrality_full(&pg, &labels);

    Ok(MfOutput {
        labels,
        similarity_matrix,
        sim_to_dist: mf_config.sim_to_dist,
        nppmi_matrix,
        raw_counts,
        ppmi_matrix,
        n,
        centrality,
    })
}

fn prepare_slices(config: &MfPipelineConfig) -> Result<Vec<PreparedSlice>> {
    let sources = discover_text_sources(&config.input_paths)?;
    match config.mf_config.slice_mode {
        MfSliceMode::None => {
            let raw_text = ingest_inputs(&config.input_paths)?;
            let tokens = prepare_tokens(&raw_text, &config.mf_config);
            if tokens.len() < config.mf_config.min_tokens_per_slice || tokens.is_empty() {
                return Ok(Vec::new());
            }
            Ok(vec![PreparedSlice {
                id: "slice-0".to_string(),
                label: "Corpus".to_string(),
                source_paths: config.input_paths.clone(),
                tokens,
            }])
        }
        MfSliceMode::PerFile => Ok(sources
            .into_iter()
            .enumerate()
            .filter_map(|(idx, source)| prepared_slice_from_source(idx, source, &config.mf_config))
            .collect()),
        MfSliceMode::FixedTokenBatch => prepare_fixed_token_batches(sources, &config.mf_config),
    }
}

fn prepared_slice_from_source(
    idx: usize,
    source: TextSource,
    mf_config: &crate::types::MfConfig,
) -> Option<PreparedSlice> {
    let tokens = prepare_tokens(&source.text, mf_config);
    if tokens.len() < mf_config.min_tokens_per_slice || tokens.is_empty() {
        return None;
    }

    Some(PreparedSlice {
        id: format!("slice-{idx}"),
        label: source.label,
        source_paths: vec![source.path],
        tokens,
    })
}

fn prepare_fixed_token_batches(
    sources: Vec<TextSource>,
    mf_config: &crate::types::MfConfig,
) -> Result<Vec<PreparedSlice>> {
    let slice_size = mf_config.slice_size.max(1);
    let all_tokens: Vec<Token> = sources
        .iter()
        .flat_map(|source| prepare_tokens(&source.text, mf_config))
        .collect();

    let mut slices = Vec::new();
    for (idx, chunk) in all_tokens.chunks(slice_size).enumerate() {
        if chunk.len() < mf_config.min_tokens_per_slice || chunk.is_empty() {
            continue;
        }
        slices.push(PreparedSlice {
            id: format!("batch-{idx}"),
            label: format!("Batch {}", idx + 1),
            source_paths: sources.iter().map(|source| source.path.clone()).collect(),
            tokens: chunk.to_vec(),
        });
    }

    Ok(slices)
}

fn build_shared_vocab(slices: &[PreparedSlice], min_count: u64) -> Vec<Token> {
    use std::collections::HashMap;

    let mut unigram_counts: HashMap<&str, u64> = HashMap::new();
    for slice in slices {
        for token in &slice.tokens {
            *unigram_counts.entry(token.as_str()).or_insert(0) += 1;
        }
    }

    let mut vocab: Vec<Token> = unigram_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_count)
        .map(|(word, _)| Token(word.to_string()))
        .collect();
    vocab.sort_by(|a, b| a.0.cmp(&b.0));
    vocab
}

fn prepare_tokens(text: &str, mf_config: &crate::types::MfConfig) -> Vec<Token> {
    let normalized = normalize_text(text, mf_config.unicode_normalize);
    let tokens = tokenize(&normalized);
    filter_stopwords(tokens, &mf_config.language)
}
