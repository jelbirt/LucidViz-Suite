//! AlignSpace pipeline end-to-end test — Phase 2.
//!
//! Reads sample_adjacency.csv (5×5 symmetric matrix), runs the full pipeline
//! with MdsConfig::Auto, ProcrustesMode::None, normalize=true, and validates
//! the output.

use as_pipeline::{
    pipeline::run_pipeline,
    types::{
        AsPipelineInput, CentralityMode, CentralityState, MdsConfig, MdsDimMode, NormalizationMode,
        ProcrustesMode,
    },
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
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_pipeline(&input).expect("pipeline failed");

    // One time step → one set of coordinates, centralities, SE matrices.
    assert_eq!(result.coordinates.len(), 1);
    assert_eq!(result.centralities.len(), 1);
    assert_eq!(result.distance_matrices.len(), 1);

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
    let centrality = match &result.centralities[0] {
        CentralityState::Computed(report) => report,
        CentralityState::Unavailable { .. } => panic!("expected computed adjacency centrality"),
    };
    assert_eq!(centrality.labels.len(), n);
    assert_eq!(centrality.degree.len(), n);
    assert_eq!(centrality.betweenness.len(), n);

    // SE matrix is symmetric and has zero diagonal.
    let se = &result.distance_matrices[0];
    assert_eq!(se.n, n);
    for i in 0..n {
        assert!(se.get(i, i) < 1e-12, "SE diagonal non-zero at {}", i);
        for j in 0..n {
            let diff = (se.get(i, j) - se.get(j, i)).abs();
            assert!(diff < 1e-12, "SE asymmetric at ({},{})", i, j);
        }
    }

    // LV dataset has one sheet with n rows.
    let lv = &result.lv_dataset;
    assert_eq!(lv.sheets.len(), 1);
    assert_eq!(lv.sheets[0].rows.len(), n);
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
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_pipeline(&input).expect("pipeline failed (two time steps)");

    assert_eq!(result.coordinates.len(), 2);
    assert_eq!(result.procrustes.len(), 2);

    // First Procrustes result is identity (residual = 0).
    let p0 = &result.procrustes[0];
    assert!((p0.scale - 1.0).abs() < 1e-10, "scale0={}", p0.scale);
    assert!(p0.residual < 1e-10, "residual0={}", p0.residual);

    for (idx, state) in result.centralities.iter().enumerate() {
        match state {
            CentralityState::Computed(report) => assert_eq!(report.labels, labels),
            CentralityState::Unavailable { .. } => {
                panic!("timestep {idx} centrality unexpectedly unavailable")
            }
        }
    }
}

#[test]
fn test_as_pipeline_optimal_choice_runs_and_preserves_labels() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let adj1 = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.2, 1.0, 0.0, 0.2, 0.2, 0.2, 0.0])
        .expect("adj1 shape");
    let adj2 = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.0])
        .expect("adj2 shape");
    let adj3 = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.8, 1.0, 0.0, 0.8, 0.8, 0.8, 0.0])
        .expect("adj3 shape");

    let input = AsPipelineInput {
        datasets: vec![
            ("T1".to_string(), adj1),
            ("T2".to_string(), adj2),
            ("T3".to_string(), adj3),
        ],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::OptimalChoice,
        mds_dims: MdsDimMode::Visual,
        normalize: false,
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_pipeline(&input).expect("optimal choice pipeline failed");

    assert_eq!(result.coordinates.len(), 3);
    assert_eq!(result.procrustes.len(), 3);
    assert_eq!(
        result
            .procrustes
            .iter()
            .filter(|r| r.residual.abs() < 1e-12)
            .count(),
        1
    );
    for coords in &result.coordinates {
        assert_eq!(coords.labels, vec!["a", "b", "c"]);
        assert!(coords.data.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_as_pipeline_3d_export_preserves_z_coordinates() {
    let csv_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_adjacency.csv"
    );

    let (labels, adj) = parse_adjacency_csv(csv_path);
    let input = AsPipelineInput {
        datasets: vec![("T1".to_string(), adj)],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: false,
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    };

    let result = run_pipeline(&input).expect("3d pipeline failed");
    let coords = &result.coordinates[0];
    let rows = &result.lv_dataset.sheets[0].rows;

    assert_eq!(coords.dims, 3);
    assert_eq!(rows.len(), coords.n);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.label, coords.labels[i]);
        assert!((row.z - coords.data[i * coords.dims + 2]).abs() < 1e-12);
    }
}

#[test]
fn test_as_pipeline_global_normalization_preserves_relative_scale() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let adj1 = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.2, 1.0, 0.0, 0.2, 0.2, 0.2, 0.0])
        .expect("adj1 shape");
    let adj2 = Array2::from_shape_vec((3, 3), vec![0.0, 10.0, 0.2, 10.0, 0.0, 0.2, 0.2, 0.2, 0.0])
        .expect("adj2 shape");

    let result = run_pipeline(&AsPipelineInput {
        datasets: vec![("T1".to_string(), adj1), ("T2".to_string(), adj2)],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Visual,
        normalize: true,
        normalization_mode: NormalizationMode::Global,
        target_range: 2.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    })
    .expect("global normalization pipeline failed");

    let max0 = result.coordinates[0]
        .data
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);
    let max1 = result.coordinates[1]
        .data
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    assert!(max1 <= 2.0 + 1e-10);
    assert!(
        max0 < max1,
        "global normalization should preserve relative scale"
    );
}

#[test]
fn test_as_pipeline_time_series_anchored_uses_first_slice_as_reference() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.2, 1.0, 0.0, 0.2, 0.2, 0.2, 0.0])
        .expect("adj shape");

    let result = run_pipeline(&AsPipelineInput {
        datasets: vec![
            ("T1".to_string(), adj.clone()),
            ("T2".to_string(), adj.clone()),
            ("T3".to_string(), adj),
        ],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::TimeSeriesAnchored,
        mds_dims: MdsDimMode::Visual,
        normalize: false,
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    })
    .expect("anchored time series pipeline failed");

    assert_eq!(result.procrustes.len(), 3);
    assert!(result.procrustes[0].residual.abs() < 1e-12);
    assert!(result.procrustes[1].residual.abs() < 1e-12);
    assert!(result.procrustes[2].residual.abs() < 1e-12);
}

#[test]
fn test_as_pipeline_preserves_directed_edges_in_lv_output() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.25, 0.0, 0.0])
        .expect("adj shape");

    let result = run_pipeline(&AsPipelineInput {
        datasets: vec![("T1".to_string(), adj)],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Visual,
        normalize: false,
        normalization_mode: NormalizationMode::Independent,
        target_range: 1.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::UndirectedLegacy,
    })
    .expect("directed pipeline failed");

    let edges = &result.lv_dataset.sheets[0].edges;
    assert_eq!(edges.len(), 3);
    assert_eq!(edges[0].from, "a");
    assert_eq!(edges[0].to, "b");
    assert_eq!(edges[1].from, "b");
    assert_eq!(edges[1].to, "c");
    assert_eq!(edges[2].from, "c");
    assert_eq!(edges[2].to, "a");
}

#[test]
fn test_as_pipeline_gpa_alignment_converges() {
    let adj_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/sample_adjacency.csv"
    );
    let (labels, adj_base) = parse_adjacency_csv(adj_path);
    let n = labels.len();

    // Create three perturbed copies.
    let mut adj_2 = adj_base.clone();
    let mut adj_3 = adj_base.clone();
    for i in 0..n {
        for j in (i + 1)..n {
            adj_2[[i, j]] += 0.03 * (i as f64);
            adj_2[[j, i]] = adj_2[[i, j]];
            adj_3[[i, j]] += 0.05 * (j as f64);
            adj_3[[j, i]] = adj_3[[i, j]];
        }
    }

    let result = run_pipeline(&AsPipelineInput {
        datasets: vec![
            ("T1".to_string(), adj_base),
            ("T2".to_string(), adj_2),
            ("T3".to_string(), adj_3),
        ],
        labels,
        mds_config: MdsConfig::Classical,
        procrustes_mode: ProcrustesMode::GPA,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: true,
        normalization_mode: NormalizationMode::Global,
        target_range: 300.0,
        procrustes_scale: true,
        centrality_mode: CentralityMode::UndirectedLegacy,
    })
    .expect("GPA pipeline failed");

    assert_eq!(result.coordinates.len(), 3);
    assert_eq!(result.procrustes.len(), 3);

    // All residuals should be finite and non-negative.
    for (i, proc) in result.procrustes.iter().enumerate() {
        assert!(
            proc.residual.is_finite() && proc.residual >= 0.0,
            "GPA result {i} residual={} invalid",
            proc.residual
        );
    }

    // All coordinates should be finite.
    for (i, coords) in result.coordinates.iter().enumerate() {
        assert_eq!(coords.dims, 3, "GPA result {i} should have 3 dims");
        for &v in &coords.data {
            assert!(v.is_finite(), "GPA result {i} has non-finite coordinate");
        }
    }
}
