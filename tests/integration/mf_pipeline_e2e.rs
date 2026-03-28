//! MatrixForge pipeline end-to-end test — Phase 3.
//!
//! Reads sample_corpus.txt, runs the full MF pipeline, and validates output.

use mf_pipeline::{
    pipeline::{run_mf_pipeline, run_mf_series_pipeline},
    types::{MfConfig, MfPipelineConfig, MfSeriesOutput, MfSliceMode},
};
use std::io::Write;

use anyhow::Result;
use as_pipeline::pipeline::mf_output_to_distance_matrix;
use as_pipeline::types::{
    AsDistancePipelineInput, CentralityMode, MdsConfig, MdsDimMode, NormalizationMode,
    ProcrustesMode,
};

/// Options for converting MF series output to AS distance pipeline input.
#[derive(Debug, Clone)]
struct MfSeriesAsInputOptions {
    mds_config: MdsConfig,
    procrustes_mode: ProcrustesMode,
    mds_dims: MdsDimMode,
    normalize: bool,
    normalization_mode: NormalizationMode,
    target_range: f64,
    procrustes_scale: bool,
    centrality_mode: CentralityMode,
}

/// Local bridge function for testing: converts MF series output to AS input.
fn mf_series_output_to_as_input(
    output: &MfSeriesOutput,
    options: MfSeriesAsInputOptions,
) -> Result<AsDistancePipelineInput> {
    output.validate_for_as_input()?;
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
        centrality_mode: options.centrality_mode,
    })
}

fn corpus_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_corpus.txt"
    ))
}

fn temp_corpus(content: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::with_suffix(".txt").expect("tempfile failed");
    file.write_all(content.as_bytes()).expect("write failed");
    file
}

fn temp_corpus_dir(files: &[(&str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir failed");
    for (name, content) in files {
        std::fs::write(dir.path().join(name), content).expect("write failed");
    }
    dir
}

#[test]
fn test_mf_pipeline_e2e_basic() {
    let cfg = MfPipelineConfig {
        input_paths: vec![corpus_path()],
        output_dir: None,
        mf_config: MfConfig {
            window_size: 3,
            slide_rate: 1,
            use_pmi: true,
            min_count: 2,
            min_pmi: 0.0,
            language: "en".to_string(),
            unicode_normalize: true,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    };

    let output = run_mf_pipeline(&cfg).expect("MF pipeline failed");

    let n = output.n;
    assert!(n > 0, "vocabulary should be non-empty");

    // similarity_matrix must be n×n
    assert_eq!(
        output.similarity_matrix.len(),
        n * n,
        "similarity_matrix length mismatch"
    );

    // All NPPMI values in [0, 1]
    for (i, &v) in output.similarity_matrix.iter().enumerate() {
        assert!(!v.is_nan(), "NaN at index {i} in similarity_matrix");
        assert!(
            v >= 0.0 && v <= 1.0 + 1e-9,
            "value {v} out of [0,1] at index {i}"
        );
    }

    // Centrality report has correct length
    assert_eq!(output.centrality.labels.len(), n);
    assert_eq!(output.centrality.degree.len(), n);
    assert_eq!(output.centrality.betweenness.len(), n);
}

#[test]
fn test_mf_pipeline_e2e_json_roundtrip() {
    let cfg = MfPipelineConfig {
        input_paths: vec![corpus_path()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 3,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    };

    let output = run_mf_pipeline(&cfg).expect("pipeline failed");

    // JSON serialize/deserialize round-trip
    let json = serde_json::to_string(&output).expect("serialize failed");
    let back: mf_pipeline::types::MfOutput =
        serde_json::from_str(&json).expect("deserialize failed");

    assert_eq!(back.n, output.n);
    assert_eq!(back.labels, output.labels);
    assert_eq!(back.similarity_matrix.len(), output.similarity_matrix.len());
    assert_eq!(back.nppmi_matrix.len(), output.nppmi_matrix.len());
    assert_eq!(back.sim_to_dist, output.sim_to_dist);
}

#[test]
fn test_mf_pipeline_e2e_as_bridge() {
    // Verify the MF output can be passed through the AS bridge and produce
    // valid MDS coordinates.
    use as_pipeline::mds::run_mds;
    use as_pipeline::pipeline::mf_output_to_distance_matrix;
    use as_pipeline::types::{MdsConfig, MdsDimMode};

    let cfg = MfPipelineConfig {
        input_paths: vec![corpus_path()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 3,
            sim_to_dist: mf_pipeline::types::SimToDistMethod::Cosine,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    };

    let output = run_mf_pipeline(&cfg).expect("pipeline failed");
    let n = output.n;

    if n < 2 {
        // Not enough vocabulary for MDS — skip.
        return;
    }

    let se = mf_output_to_distance_matrix(
        output.labels.clone(),
        &output.similarity_matrix,
        n,
        output.sim_to_dist,
    )
    .expect("distance matrix conversion failed");

    let coords = run_mds(&se, &MdsConfig::Classical, MdsDimMode::Visual).expect("MDS failed");

    assert_eq!(coords.n, n);
    assert!(coords.stress.is_finite(), "MDS stress should be finite");
    assert_eq!(coords.data.len(), n * 2);
}

#[test]
fn test_mf_pipeline_e2e_xlsx_output() {
    // Run with xlsx output and verify the file can be read back by calamine.
    let tmp_dir = tempfile::tempdir().expect("tempdir failed");

    let cfg = MfPipelineConfig {
        input_paths: vec![corpus_path()],
        output_dir: Some(tmp_dir.path().to_path_buf()),
        mf_config: MfConfig {
            min_count: 3,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: true,
    };

    run_mf_pipeline(&cfg).expect("pipeline with xlsx output failed");

    let xlsx_path = tmp_dir.path().join("mf_output.xlsx");
    assert!(xlsx_path.exists(), "mf_output.xlsx not found");

    // Try to open with calamine.
    use calamine::{open_workbook, Reader, Xlsx};
    let wb: Xlsx<_> = open_workbook(&xlsx_path).expect("calamine failed to open xlsx");
    let sheet_names = wb.sheet_names().to_vec();
    assert!(
        sheet_names.contains(&"Vocabulary".to_string()),
        "missing Vocabulary sheet"
    );
    assert!(
        sheet_names.contains(&"NPPMI Matrix".to_string()),
        "missing NPPMI Matrix sheet"
    );
    assert!(
        sheet_names.contains(&"PPMI Matrix".to_string()),
        "missing PPMI Matrix sheet"
    );
}

#[test]
fn test_mf_pipeline_honors_use_pmi_toggle() {
    let corpus = temp_corpus("alpha beta alpha beta gamma alpha beta");

    let base_cfg = MfPipelineConfig {
        input_paths: vec![corpus.path().to_path_buf()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    };

    let pmi_output = run_mf_pipeline(&base_cfg).expect("PMI pipeline failed");

    let raw_cfg = MfPipelineConfig {
        mf_config: MfConfig {
            use_pmi: false,
            min_count: 1,
            ..MfConfig::default()
        },
        ..base_cfg.clone()
    };
    let raw_output = run_mf_pipeline(&raw_cfg).expect("raw-count pipeline failed");

    assert_eq!(pmi_output.similarity_matrix, pmi_output.nppmi_matrix);
    assert_ne!(raw_output.similarity_matrix, raw_output.nppmi_matrix);
    assert!(raw_output
        .similarity_matrix
        .iter()
        .all(|v| (0.0..=1.0).contains(v)));
}

#[test]
fn test_mf_pipeline_honors_unicode_normalize_toggle() {
    let corpus = temp_corpus("Cafe\u{0301} cafe\u{0301} caf\u{00e9} caf\u{00e9}");

    let normalized = run_mf_pipeline(&MfPipelineConfig {
        input_paths: vec![corpus.path().to_path_buf()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            unicode_normalize: true,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    })
    .expect("normalized pipeline failed");

    let raw = run_mf_pipeline(&MfPipelineConfig {
        input_paths: vec![corpus.path().to_path_buf()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            unicode_normalize: false,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    })
    .expect("raw pipeline failed");

    assert!(normalized.labels.iter().any(|label| label == "café"));
    assert!(raw.labels.iter().any(|label| label == "cafe"));
    assert_ne!(normalized.labels, raw.labels);
}

#[test]
fn test_mf_series_pipeline_per_file_shared_vocab() {
    let dir = temp_corpus_dir(&[
        ("a.txt", "alpha beta alpha gamma"),
        ("b.txt", "beta gamma beta delta"),
    ]);

    let output = run_mf_series_pipeline(&MfPipelineConfig {
        input_paths: vec![dir.path().to_path_buf()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            slice_mode: MfSliceMode::PerFile,
            shared_vocabulary: true,
            min_tokens_per_slice: 1,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    })
    .expect("series pipeline failed");

    assert_eq!(output.slices.len(), 2);
    assert_eq!(output.slices[0].label, "a");
    assert_eq!(output.slices[1].label, "b");
    assert_eq!(output.labels, output.slices[0].output.labels);
    assert_eq!(output.labels, output.slices[1].output.labels);
}

#[test]
fn test_mf_series_pipeline_bridges_to_as_distance_pipeline() {
    use as_pipeline::pipeline::run_distance_pipeline;
    use as_pipeline::types::{
        CentralityState, MdsConfig, MdsDimMode, NormalizationMode, ProcrustesMode,
    };

    let dir = temp_corpus_dir(&[
        ("a.txt", "alpha beta alpha gamma"),
        ("b.txt", "alpha gamma gamma beta"),
    ]);

    let series = run_mf_series_pipeline(&MfPipelineConfig {
        input_paths: vec![dir.path().to_path_buf()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            slice_mode: MfSliceMode::PerFile,
            shared_vocabulary: true,
            min_tokens_per_slice: 1,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    })
    .expect("series pipeline failed");

    let input = mf_series_output_to_as_input(
        &series,
        MfSeriesAsInputOptions {
            mds_config: MdsConfig::Classical,
            procrustes_mode: ProcrustesMode::TimeSeries,
            mds_dims: MdsDimMode::Visual,
            normalize: true,
            normalization_mode: NormalizationMode::Independent,
            target_range: 300.0,
            procrustes_scale: true,
            centrality_mode: as_pipeline::types::CentralityMode::Directed,
        },
    )
    .expect("series conversion failed");

    let result = run_distance_pipeline(&input).expect("distance pipeline failed");
    assert_eq!(result.coordinates.len(), 2);
    assert_eq!(result.etv_dataset.sheets.len(), 2);
    assert_eq!(result.etv_dataset.all_labels, series.labels);
    assert!(matches!(
        &result.centralities[0],
        CentralityState::Unavailable { .. }
    ));
}

#[test]
fn test_single_output_and_single_slice_series_produce_same_distance_dataset() {
    use as_pipeline::pipeline::{mf_output_to_distance_matrix, run_distance_pipeline};
    use as_pipeline::types::{MdsConfig, MdsDimMode, NormalizationMode, ProcrustesMode};
    use mf_pipeline::types::{MfSeriesOutput, MfSlice};

    let output = run_mf_pipeline(&MfPipelineConfig {
        input_paths: vec![corpus_path()],
        output_dir: None,
        mf_config: MfConfig {
            min_count: 1,
            ..MfConfig::default()
        },
        write_json: false,
        write_xlsx: false,
    })
    .expect("single pipeline failed");

    let single_input = as_pipeline::types::AsDistancePipelineInput {
        datasets: vec![(
            "MatrixForge".to_string(),
            mf_output_to_distance_matrix(
                output.labels.clone(),
                &output.similarity_matrix,
                output.n,
                output.sim_to_dist,
            )
            .expect("distance matrix conversion failed"),
        )],
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: true,
        normalization_mode: NormalizationMode::Independent,
        target_range: 300.0,
        procrustes_scale: true,
        centrality_mode: as_pipeline::types::CentralityMode::Directed,
    };
    let single_result =
        run_distance_pipeline(&single_input).expect("single distance pipeline failed");

    let series = MfSeriesOutput {
        labels: output.labels.clone(),
        sim_to_dist: output.sim_to_dist,
        slices: vec![MfSlice {
            id: "slice-1".to_string(),
            label: "MatrixForge".to_string(),
            order: 0,
            source_paths: vec![corpus_path()],
            token_count: 0,
            output,
        }],
    };
    let series_input = mf_series_output_to_as_input(
        &series,
        MfSeriesAsInputOptions {
            mds_config: MdsConfig::Classical,
            procrustes_mode: ProcrustesMode::None,
            mds_dims: MdsDimMode::Fixed(3),
            normalize: true,
            normalization_mode: NormalizationMode::Independent,
            target_range: 300.0,
            procrustes_scale: true,
            centrality_mode: as_pipeline::types::CentralityMode::Directed,
        },
    )
    .expect("series conversion failed");
    let series_result =
        run_distance_pipeline(&series_input).expect("series distance pipeline failed");

    assert_eq!(
        single_result.etv_dataset.all_labels,
        series_result.etv_dataset.all_labels
    );
    assert_eq!(single_result.etv_dataset.sheets.len(), 1);
    assert_eq!(
        single_result.etv_dataset.sheets,
        series_result.etv_dataset.sheets
    );
}
