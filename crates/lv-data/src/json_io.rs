use std::path::Path;

use crate::{DataError, EtvDataset};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise an [`EtvDataset`] to a pretty-printed JSON file.
pub fn write_etv_json(dataset: &EtvDataset, path: &Path) -> Result<(), DataError> {
    let json = serde_json::to_string_pretty(dataset)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Deserialise an [`EtvDataset`] from a JSON file.
pub fn read_etv_json(path: &Path) -> Result<EtvDataset, DataError> {
    let bytes = std::fs::read(path)?;
    let dataset: EtvDataset = serde_json::from_slice(&bytes)?;
    Ok(dataset)
}

/// Serialise an [`EtvDataset`] to an in-memory JSON byte vector.
pub fn write_etv_json_bytes(dataset: &EtvDataset) -> Result<Vec<u8>, DataError> {
    let json = serde_json::to_vec_pretty(dataset)?;
    Ok(json)
}

/// Deserialise an [`EtvDataset`] from an in-memory JSON byte slice.
pub fn read_etv_json_bytes(bytes: &[u8]) -> Result<EtvDataset, DataError> {
    let dataset: EtvDataset = serde_json::from_slice(bytes)?;
    Ok(dataset)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EtvDataset, EtvRow, EtvSheet, ShapeKind};

    fn minimal_dataset() -> EtvDataset {
        let row = EtvRow {
            label: "NodeA".into(),
            x: 0.5,
            y: -0.5,
            z: 1.0,
            size: 2.0,
            size_alpha: 0.1,
            spin_x: 10.0,
            spin_y: 0.0,
            spin_z: 0.0,
            shape: ShapeKind::Torus,
            color_r: 0.3,
            color_g: 0.6,
            color_b: 0.9,
            note: 72,
            instrument: 5,
            channel: 2,
            velocity: 100,
            cluster_value: 3.0,
            beats: 4,
        };
        EtvDataset {
            source_path: None,
            sheets: vec![EtvSheet {
                name: "T0".into(),
                sheet_index: 0,
                rows: vec![row],
                edges: vec![],
            }],
            all_labels: vec!["NodeA".into()],
        }
    }

    #[test]
    fn json_roundtrip_bytes() {
        let original = minimal_dataset();
        let bytes = write_etv_json_bytes(&original).expect("write");
        let recovered = read_etv_json_bytes(&bytes).expect("read");

        assert_eq!(recovered.sheets.len(), 1);
        let row = &recovered.sheets[0].rows[0];
        assert_eq!(row.label, "NodeA");
        assert_eq!(row.shape, ShapeKind::Torus);
        assert!((row.x - 0.5).abs() < 1e-12);
        assert_eq!(row.beats, 4);
        assert_eq!(row.velocity, 100);
    }

    #[test]
    fn json_output_is_pretty() {
        let ds = minimal_dataset();
        let bytes = write_etv_json_bytes(&ds).expect("write");
        let s = std::str::from_utf8(&bytes).expect("utf8");
        // Pretty JSON must contain newlines and indentation
        assert!(s.contains('\n'));
        assert!(s.contains("  "));
    }
}
