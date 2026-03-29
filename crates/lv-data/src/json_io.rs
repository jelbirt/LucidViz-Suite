use std::path::Path;

use crate::{validate_dataset, DataError, LvDataset};

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise an [`LvDataset`] to a pretty-printed JSON file.
pub fn write_lv_json(dataset: &LvDataset, path: &Path) -> Result<(), DataError> {
    let json = serde_json::to_string_pretty(dataset)?;
    crate::io_util::atomic_write(path, json.as_bytes())?;
    Ok(())
}

/// Deserialise an [`LvDataset`] from a JSON file.
pub fn read_lv_json(path: &Path) -> Result<LvDataset, DataError> {
    let bytes = crate::io_util::read_bounded_file(path, MAX_JSON_BYTES)?;
    let mut dataset: LvDataset = serde_json::from_slice(&bytes)?;
    dataset.canonicalize_all_labels();
    validate_dataset(&dataset)?;
    Ok(dataset)
}

/// Serialise an [`LvDataset`] to an in-memory JSON byte vector.
pub fn write_lv_json_bytes(dataset: &LvDataset) -> Result<Vec<u8>, DataError> {
    let json = serde_json::to_vec_pretty(dataset)?;
    Ok(json)
}

/// Deserialise an [`LvDataset`] from an in-memory JSON byte slice.
pub fn read_lv_json_bytes(bytes: &[u8]) -> Result<LvDataset, DataError> {
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(DataError::FileTooLarge {
            path: "<memory>".into(),
            bytes: bytes.len() as u64,
            limit: MAX_JSON_BYTES,
        });
    }
    let mut dataset: LvDataset = serde_json::from_slice(bytes)?;
    dataset.canonicalize_all_labels();
    validate_dataset(&dataset)?;
    Ok(dataset)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LvDataset, LvRow, LvSheet, ShapeKind};

    fn minimal_dataset() -> LvDataset {
        let row = LvRow {
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
        LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
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
        let bytes = write_lv_json_bytes(&original).expect("write");
        let recovered = read_lv_json_bytes(&bytes).expect("read");

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
        let bytes = write_lv_json_bytes(&ds).expect("write");
        let s = std::str::from_utf8(&bytes).expect("utf8");
        // Pretty JSON must contain newlines and indentation
        assert!(s.contains('\n'));
        assert!(s.contains("  "));
    }

    #[test]
    fn json_read_canonicalizes_all_labels() {
        let json = br#"{
          "source_path": null,
          "sheets": [
            {
              "name": "T0",
              "sheet_index": 0,
              "rows": [
                {
                  "label": "NodeA",
                  "x": 0.0,
                  "y": 0.0,
                  "z": 0.0,
                  "size": 1.0,
                  "size_alpha": 0.0,
                  "spin_x": 0.0,
                  "spin_y": 0.0,
                  "spin_z": 0.0,
                  "shape": "Sphere",
                  "color_r": 1.0,
                  "color_g": 1.0,
                  "color_b": 1.0,
                  "note": 60,
                  "instrument": 0,
                  "channel": 0,
                  "velocity": 64,
                  "cluster_value": 0.0,
                  "beats": 0
                }
              ],
              "edges": []
            }
          ],
          "all_labels": ["ghost", "NodeA"]
        }"#;

        let recovered = read_lv_json_bytes(json).expect("read");
        assert_eq!(recovered.all_labels, vec!["NodeA"]);
    }

    #[test]
    fn json_read_rejects_oversized_memory_input() {
        let bytes = vec![b' '; (MAX_JSON_BYTES + 1) as usize];
        let err = read_lv_json_bytes(&bytes).expect_err("oversized json should fail");
        assert!(matches!(err, DataError::FileTooLarge { .. }));
    }
}
