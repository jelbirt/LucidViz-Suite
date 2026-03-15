//! Coordinate normalization: scale all coordinates to fit within ±target_range.

use crate::types::MdsCoordinates;

/// Scale coordinates so that the maximum absolute value across all dimensions
/// equals `target_range`.  No-op if all coordinates are zero.
pub fn normalize_coordinates(coords: &mut MdsCoordinates, target_range: f64) {
    let max_abs = coords.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    if max_abs < 1e-15 {
        return; // nothing to scale
    }

    let scale = target_range / max_abs;
    for v in coords.data.iter_mut() {
        *v *= scale;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MdsAlgorithm;

    fn make_coords(data: Vec<f64>, dims: usize) -> MdsCoordinates {
        let n = data.len() / dims;
        let labels: Vec<String> = (0..n).map(|i| format!("n{}", i)).collect();
        MdsCoordinates::new(labels, data, dims, 0.0, MdsAlgorithm::Classical)
    }

    #[test]
    fn test_normalize_within_range() {
        let mut coords = make_coords(vec![-3.0, 1.5, 2.0, -0.5], 2);
        normalize_coordinates(&mut coords, 1.0);
        let max_abs = coords.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!((max_abs - 1.0).abs() < 1e-10, "max_abs={}", max_abs);
    }

    #[test]
    fn test_normalize_zero_noop() {
        let mut coords = make_coords(vec![0.0, 0.0, 0.0, 0.0], 2);
        normalize_coordinates(&mut coords, 1.0);
        assert!(coords.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_custom_range() {
        let mut coords = make_coords(vec![1.0, 2.0, 3.0, 4.0], 2);
        normalize_coordinates(&mut coords, 10.0);
        let max_abs = coords.data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!((max_abs - 10.0).abs() < 1e-10, "max_abs={}", max_abs);
    }
}
