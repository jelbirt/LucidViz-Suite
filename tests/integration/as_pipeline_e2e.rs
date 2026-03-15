//! AlignSpace pipeline end-to-end test — Phase 2.
//!
//! Reads sample_adjacency.csv (5×5 symmetric matrix), runs the full pipeline
//! with MdsConfig::Auto, ProcrustesMode::None, normalize=true, and validates
//! the output.

use as_pipeline::{
    pipeline::run_pipeline,
    types::{AsPipelineInput, MdsConfig, MdsDimMode, ProcrustesMode},
};
use ndarray::Array2;
use std::io::{BufRead, BufReader};

/// Parse sample_adjacency.csv into (labels, Array2<f64>).
///
/// CSV format: first row = "label,node1,..."; subsequent rows = "nodeN,val,...".
fn parse_adjacency_csv(path: &str) -> (Vec<String>, Array2<f64>) {
    let file = std::fs::File::open(path).expect("sample_adjacency.csv not found");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|l| l.unwrap());

    // Header: label,word1,word2,...
    let header = lines.next().expect("empty file");
    let labels: Vec<String> = header
        .split(',')
        .skip(1)
        .map(|s| s.trim().to_string())
        .collect();
    let n = labels.len();

    let mut data = vec![0.0f64; n * n];
    for (row_idx, line) in lines.enumerate() {
        if line.trim().is_empty() {
            break;
        }
        let parts: Vec<&str> = line.split(',').collect();
        for col_idx in 0..n {
            data[row_idx * n + col_idx] = parts[col_idx + 1].trim().parse::<f64>().unwrap_or(0.0);
        }
    }

    let adj = Array2::from_shape_vec((n, n), data).expect("shape mismatch");
    (labels, adj)
}

#[test]
fn test_as_pipeline_e2e_single_timestep() {
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_adjacency.csv"
    );

    let (labels, adj) = parse_adjacency_csv(csv_path);
    let n = labels.len(); // 5

    let input = AsPipelineInput {
        datasets: vec![("T1".to_string(), adj)],
        labels: labels.clone(),
        mds_config: MdsConfig::Auto,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Visual, // 2D
        normalize: true,
        target_range: 1.0,
        procrustes_scale: false,
    };

    let result = run_pipeline(&input).expect("pipeline failed");

    // One time step → one set of coordinates, centralities, SE matrices.
    assert_eq!(result.coordinates.len(), 1);
    assert_eq!(result.centralities.len(), 1);
    assert_eq!(result.se_matrices.len(), 1);

    let coords = &result.coordinates[0];
    assert_eq!(coords.n, n);
    assert_eq!(coords.dims, 2);
    assert_eq!(coords.data.len(), n * 2);

    // After normalization, max_abs should be ≤ 1.0.
    let max_abs = coords.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    assert!(
        max_abs <= 1.0 + 1e-10,
        "max_abs={} exceeds target_range=1.0",
        max_abs
    );

    // Stress should be reasonable for a 5-node dataset.
    assert!(coords.stress < 0.5, "stress={} too high", coords.stress);

    // Centrality report has correct number of labels.
    let centrality = &result.centralities[0];
    assert_eq!(centrality.labels.len(), n);
    assert_eq!(centrality.degree.len(), n);
    assert_eq!(centrality.betweenness.len(), n);

    // SE matrix is symmetric and has zero diagonal.
    let se = &result.se_matrices[0];
    assert_eq!(se.n, n);
    for i in 0..n {
        assert!(se.get(i, i) < 1e-12, "SE diagonal non-zero at {}", i);
        for j in 0..n {
            let diff = (se.get(i, j) - se.get(j, i)).abs();
            assert!(diff < 1e-12, "SE asymmetric at ({},{})", i, j);
        }
    }

    // ETV dataset has one sheet with n rows.
    let etv = &result.etv_dataset;
    assert_eq!(etv.sheets.len(), 1);
    assert_eq!(etv.sheets[0].rows.len(), n);
}

#[test]
fn test_as_pipeline_e2e_two_timesteps() {
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_adjacency.csv"
    );

    let (labels, adj) = parse_adjacency_csv(csv_path);

    // Slightly perturb the second matrix.
    let mut adj2 = adj.clone();
    adj2[[0, 1]] *= 0.9;
    adj2[[1, 0]] *= 0.9;

    let input = AsPipelineInput {
        datasets: vec![("T1".to_string(), adj), ("T2".to_string(), adj2)],
        labels: labels.clone(),
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::TimeSeries,
        mds_dims: MdsDimMode::Visual,
        normalize: true,
        target_range: 1.0,
        procrustes_scale: false,
    };

    let result = run_pipeline(&input).expect("pipeline failed (two time steps)");

    assert_eq!(result.coordinates.len(), 2);
    assert_eq!(result.procrustes.len(), 2);

    // First Procrustes result is identity (residual = 0).
    let p0 = &result.procrustes[0];
    assert!((p0.scale - 1.0).abs() < 1e-10, "scale0={}", p0.scale);
    assert!(p0.residual < 1e-10, "residual0={}", p0.residual);
}
