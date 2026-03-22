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

/// Scale every coordinate set in a series with one shared factor so the global
/// maximum absolute value equals `target_range`. No-op if the whole series is zero.
pub fn normalize_coordinate_series(coords: &mut [MdsCoordinates], target_range: f64) {
    let max_abs = coords
        .iter()
        .flat_map(|c| c.data.iter())
        .map(|v| v.abs())
        .fold(0.0f64, f64::max);

    if max_abs < 1e-15 {
        return;
    }

    let scale = target_range / max_abs;
    for coord_set in coords.iter_mut() {
        for value in &mut coord_set.data {
            *value *= scale;
        }
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
            .expect("test coordinates should build")
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

    #[test]
    fn test_global_normalize_preserves_relative_scale_between_slices() {
        let mut coords = vec![
            make_coords(vec![1.0, 0.0, 0.0, 1.0], 2),
            make_coords(vec![4.0, 0.0, 0.0, 4.0], 2),
        ];

        normalize_coordinate_series(&mut coords, 2.0);

        let first_max = coords[0]
            .data
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        let second_max = coords[1]
            .data
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);

        assert!((first_max - 0.5).abs() < 1e-10, "first_max={first_max}");
        assert!((second_max - 2.0).abs() < 1e-10, "second_max={second_max}");
    }
}
