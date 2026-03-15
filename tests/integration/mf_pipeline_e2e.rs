//! MatrixForge pipeline end-to-end test — Phase 3.
//!
//! Reads sample_corpus.txt, runs the full MF pipeline, and validates output.

use mf_pipeline::{
    pipeline::run_mf_pipeline,
    types::{MfConfig, MfPipelineConfig},
};

fn corpus_path() -> std::path::PathBuf {
    std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_corpus.txt"
    ))
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
}

#[test]
fn test_mf_pipeline_e2e_as_bridge() {
    // Verify the MF output can be passed through the AS bridge and produce
    // valid MDS coordinates.
    use as_pipeline::mds::run_mds;
    use as_pipeline::pipeline::{mf_output_to_se_matrix, SimToDistMethod};
    use as_pipeline::types::{MdsConfig, MdsDimMode};

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
    let n = output.n;

    if n < 2 {
        // Not enough vocabulary for MDS — skip.
        return;
    }

    let se = mf_output_to_se_matrix(
        output.labels.clone(),
        &output.similarity_matrix,
        n,
        SimToDistMethod::Linear,
    );

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
        sheet_names.contains(&"PMI Matrix".to_string()),
        "missing PMI Matrix sheet"
    );
}
