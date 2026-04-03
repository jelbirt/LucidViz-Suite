use as_pipeline::procrustes::procrustes;
use as_pipeline::types::{MdsAlgorithm, MdsCoordinates};
use proptest::prelude::*;

/// Generate random MdsCoordinates with n labels and d dimensions.
fn arb_coords(n: usize, dims: usize) -> impl Strategy<Value = MdsCoordinates> {
    proptest::collection::vec(1.0..100.0_f64, n * dims).prop_map(move |data| {
        let labels: Vec<String> = (0..n).map(|i| format!("n{i}")).collect();
        MdsCoordinates::new(labels, data, dims, 0.0, MdsAlgorithm::Classical).unwrap()
    })
}

proptest! {
    #[test]
    fn procrustes_residual_non_negative(
        coords in arb_coords(5, 3),
    ) {
        let result = procrustes(&coords, &coords, true).unwrap();
        prop_assert!(result.residual >= 0.0,
            "residual should be non-negative, got {}", result.residual);
    }

    #[test]
    fn procrustes_self_alignment_low_residual(
        coords in arb_coords(5, 3),
    ) {
        // Aligning a configuration to itself should yield near-zero residual.
        let result = procrustes(&coords, &coords, true).unwrap();
        prop_assert!(result.residual < 1e-6,
            "self-alignment residual should be near zero, got {}", result.residual);
    }

    #[test]
    fn procrustes_rotation_orthogonal(
        source in arb_coords(5, 3),
        target in arb_coords(5, 3),
    ) {
        let result = procrustes(&source, &target, true).unwrap();
        let dims = source.dims.min(target.dims);
        let rot = &result.rotation;
        // R^T * R should be close to identity
        for i in 0..dims {
            for j in 0..dims {
                let mut dot = 0.0;
                for k in 0..dims {
                    dot += rot[k * dims + i] * rot[k * dims + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                prop_assert!((dot - expected).abs() < 1e-6,
                    "R^T*R[{i},{j}] = {dot}, expected {expected}");
            }
        }
    }

    #[test]
    fn procrustes_scale_positive(
        source in arb_coords(5, 3),
        target in arb_coords(5, 3),
    ) {
        let result = procrustes(&source, &target, true).unwrap();
        prop_assert!(result.scale > 0.0,
            "scale should be positive, got {}", result.scale);
    }
}
