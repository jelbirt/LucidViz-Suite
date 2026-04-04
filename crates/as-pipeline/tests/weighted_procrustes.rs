use as_pipeline::procrustes::procrustes_weighted;
use as_pipeline::types::{MdsAlgorithm, MdsCoordinates};

fn make_coords(labels: Vec<String>, data: Vec<f64>, dims: usize) -> MdsCoordinates {
    MdsCoordinates::new(labels, data, dims, 0.0, MdsAlgorithm::Classical).unwrap()
}

fn labels(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("n{i}")).collect()
}

#[test]
fn weighted_self_alignment_near_zero_residual() {
    let coords = make_coords(labels(4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2);
    let weights = vec![1.0, 2.0, 0.5, 3.0];
    let result = procrustes_weighted(&coords, &coords, true, Some(&weights)).unwrap();
    assert!(
        result.residual < 1e-6,
        "self-alignment residual with weights should be near zero, got {}",
        result.residual
    );
}

#[test]
fn weighted_rotation_is_orthogonal() {
    let source = make_coords(labels(4), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);
    // 90-degree rotation
    let target = make_coords(labels(4), vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0], 2);
    let weights = vec![1.0, 5.0, 1.0, 5.0]; // non-uniform
    let result = procrustes_weighted(&source, &target, true, Some(&weights)).unwrap();

    let dims = 2;
    let rot = &result.rotation;
    for i in 0..dims {
        for j in 0..dims {
            let mut dot = 0.0;
            for k in 0..dims {
                dot += rot[k * dims + i] * rot[k * dims + j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-6,
                "R^T*R[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn weighted_scale_is_positive() {
    let source = make_coords(labels(3), vec![0.0, 0.0, 5.0, 0.0, 0.0, 5.0], 2);
    let target = make_coords(labels(3), vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0], 2);
    let weights = vec![0.1, 10.0, 1.0];
    let result = procrustes_weighted(&source, &target, true, Some(&weights)).unwrap();
    assert!(
        result.scale > 0.0,
        "scale should be positive, got {}",
        result.scale
    );
}

#[test]
fn zero_weight_nodes_have_less_influence() {
    // Two alignments: one with uniform weights, one with node 0 down-weighted.
    // The down-weighted version should differ from uniform.
    let source = make_coords(labels(4), vec![10.0, 10.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);
    let target = make_coords(labels(4), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 2);

    let uniform = procrustes_weighted(&source, &target, true, Some(&[1.0, 1.0, 1.0, 1.0])).unwrap();
    let downweight =
        procrustes_weighted(&source, &target, true, Some(&[0.01, 1.0, 1.0, 1.0])).unwrap();

    // The residuals should differ because weighting changes the optimization
    assert!(
        (uniform.residual - downweight.residual).abs() > 1e-6
            || (uniform.scale - downweight.scale).abs() > 1e-6,
        "down-weighting a node should change the result"
    );
}

#[test]
fn none_weights_matches_unweighted() {
    let source = make_coords(labels(4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2);
    let target = make_coords(labels(4), vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0], 2);

    let unweighted = procrustes_weighted(&source, &target, true, None).unwrap();
    let explicit_uniform =
        procrustes_weighted(&source, &target, true, Some(&[1.0, 1.0, 1.0, 1.0])).unwrap();

    assert!(
        (unweighted.residual - explicit_uniform.residual).abs() < 1e-6,
        "None weights should match uniform: {} vs {}",
        unweighted.residual,
        explicit_uniform.residual
    );
}
