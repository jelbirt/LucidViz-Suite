use as_pipeline::types::{DistanceMatrix, MdsAlgorithm, MdsConfig, MdsDimMode};

/// Build a synthetic symmetric distance matrix of size n.
fn synthetic_distance_matrix(n: usize) -> DistanceMatrix {
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = ((i as f64 - j as f64).abs() + 1.0).sqrt();
            data[i * n + j] = d;
            data[j * n + i] = d;
        }
    }
    let labels: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
    DistanceMatrix::new(labels, data).unwrap()
}

#[test]
fn auto_selects_classical_below_800() {
    // n=10 should use Classical MDS via Auto.
    let dist = synthetic_distance_matrix(10);
    let result = as_pipeline::mds::run_mds(&dist, &MdsConfig::Auto, MdsDimMode::Fixed(3)).unwrap();
    assert_eq!(result.algorithm, MdsAlgorithm::Classical);
}

#[test]
fn explicit_pivot_produces_valid_coordinates() {
    // Use explicit PivotMds with small n to verify the pipeline works.
    let dist = synthetic_distance_matrix(50);
    let result = as_pipeline::mds::run_mds(
        &dist,
        &MdsConfig::PivotMds { n_pivots: 20 },
        MdsDimMode::Fixed(3),
    )
    .unwrap();
    assert_eq!(result.n, 50);
    assert_eq!(result.dims, 3);
    assert_eq!(result.algorithm, MdsAlgorithm::PivotMds);
    for val in &result.data {
        assert!(val.is_finite(), "coordinate must be finite");
    }
}

#[test]
fn auto_classical_produces_finite_coords() {
    let dist = synthetic_distance_matrix(20);
    let result = as_pipeline::mds::run_mds(&dist, &MdsConfig::Auto, MdsDimMode::Fixed(3)).unwrap();
    assert_eq!(result.n, 20);
    assert_eq!(result.dims, 3);
    for val in &result.data {
        assert!(val.is_finite());
    }
}
