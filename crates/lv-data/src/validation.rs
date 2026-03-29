use crate::{DataError, EdgeRow, LvDataset, LvRow, LvSheet};

/// A single validation failure with its location.
#[derive(Debug, Clone)]
pub struct ValidationFailure {
    pub sheet: String,
    /// 1-based row index within the object section of that sheet.
    pub row: usize,
    pub msg: String,
}

impl std::fmt::Display for ValidationFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sheet '{}' row {}: {}", self.sheet, self.row, self.msg)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row-level validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validate a single [`LvRow`].  Returns a list of failure messages
/// (empty = valid).
pub fn validate_row(row: &LvRow) -> Vec<String> {
    let mut errs: Vec<String> = Vec::new();

    if row.label.is_empty() {
        errs.push("label must not be empty".into());
    } else if row.label.len() > 255 {
        errs.push(format!("label exceeds 255 chars (got {})", row.label.len()));
    }

    if !row.x.is_finite() {
        errs.push(format!("x is not finite: {}", row.x));
    }
    if !row.y.is_finite() {
        errs.push(format!("y is not finite: {}", row.y));
    }
    if !row.z.is_finite() {
        errs.push(format!("z is not finite: {}", row.z));
    }

    if !row.size.is_finite() || row.size <= 0.0 {
        errs.push(format!("size must be > 0 and finite (got {})", row.size));
    }
    if !row.size_alpha.is_finite() || row.size_alpha < 0.0 {
        errs.push(format!(
            "size_alpha must be >= 0 and finite (got {})",
            row.size_alpha
        ));
    }

    if !row.spin_x.is_finite() {
        errs.push(format!("spin_x is not finite: {}", row.spin_x));
    }
    if !row.spin_y.is_finite() {
        errs.push(format!("spin_y is not finite: {}", row.spin_y));
    }
    if !row.spin_z.is_finite() {
        errs.push(format!("spin_z is not finite: {}", row.spin_z));
    }

    for (name, val) in [
        ("color_r", row.color_r),
        ("color_g", row.color_g),
        ("color_b", row.color_b),
    ] {
        if !(0.0..=1.0).contains(&val) {
            errs.push(format!("{name} must be in [0, 1] (got {val})"));
        }
    }

    if row.note > 127 {
        errs.push(format!("note must be in [0, 127] (got {})", row.note));
    }
    if row.instrument > 365 {
        errs.push(format!(
            "instrument must be in [0, 365] (got {})",
            row.instrument
        ));
    }
    if row.channel > 15 {
        errs.push(format!("channel must be in [0, 15] (got {})", row.channel));
    }
    if row.velocity == 0 || row.velocity > 127 {
        errs.push(format!(
            "velocity must be in [1, 127] (got {})",
            row.velocity
        ));
    }

    if !row.cluster_value.is_finite() || row.cluster_value < 0.0 {
        errs.push(format!(
            "cluster_value must be >= 0 and finite (got {})",
            row.cluster_value
        ));
    }

    errs
}

/// Validate a single [`EdgeRow`] against the set of known labels.
pub fn validate_edge(
    edge: &EdgeRow,
    labels: &std::collections::HashSet<&str>,
    sheet: &str,
) -> Vec<String> {
    let mut errs = Vec::new();
    if !labels.contains(edge.from.as_str()) {
        errs.push(format!(
            "edge 'from' label '{}' not found in sheet '{sheet}'",
            edge.from
        ));
    }
    if !labels.contains(edge.to.as_str()) {
        errs.push(format!(
            "edge 'to' label '{}' not found in sheet '{sheet}'",
            edge.to
        ));
    }
    if !edge.strength.is_finite() {
        errs.push(format!("edge strength is not finite: {}", edge.strength));
    }
    errs
}

// ─────────────────────────────────────────────────────────────────────────────
// Sheet-level validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validate all rows and edges in a single [`LvSheet`].
pub fn validate_sheet(sheet: &LvSheet) -> Vec<ValidationFailure> {
    let mut failures = Vec::new();

    let label_set: std::collections::HashSet<&str> =
        sheet.rows.iter().map(|r| r.label.as_str()).collect();

    for (i, row) in sheet.rows.iter().enumerate() {
        for msg in validate_row(row) {
            failures.push(ValidationFailure {
                sheet: sheet.name.clone(),
                row: i + 1,
                msg,
            });
        }
    }

    for (i, edge) in sheet.edges.iter().enumerate() {
        for msg in validate_edge(edge, &label_set, &sheet.name) {
            failures.push(ValidationFailure {
                sheet: sheet.name.clone(),
                row: sheet.rows.len() + i + 1,
                msg,
            });
        }
    }

    failures
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset-level validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validate an entire [`LvDataset`].
///
/// Returns `Ok(())` if all rows and edges are valid, or a [`DataError::Validation`]
/// aggregating every failure.
pub fn validate_dataset(dataset: &LvDataset) -> Result<(), DataError> {
    let mut all_failures: Vec<ValidationFailure> = Vec::new();

    let canonical_labels = dataset.canonical_all_labels();
    if dataset.all_labels != canonical_labels {
        all_failures.push(ValidationFailure {
            sheet: "<dataset>".into(),
            row: 0,
            msg: format!(
                "all_labels does not match the canonical first-seen union of sheet row labels; expected {:?}, got {:?}",
                canonical_labels, dataset.all_labels
            ),
        });
    }

    for sheet in &dataset.sheets {
        all_failures.extend(validate_sheet(sheet));
    }

    if all_failures.is_empty() {
        Ok(())
    } else {
        let messages = all_failures
            .iter()
            .map(|f| f.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        Err(DataError::Validation {
            count: all_failures.len(),
            messages,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{LvRow, ShapeKind};

    fn valid_row() -> LvRow {
        LvRow {
            label: "TestNode".into(),
            x: 1.0,
            y: -2.0,
            z: 0.5,
            size: 1.0,
            size_alpha: 0.0,
            spin_x: 0.0,
            spin_y: 0.0,
            spin_z: 0.0,
            shape: ShapeKind::Sphere,
            color_r: 0.5,
            color_g: 0.5,
            color_b: 0.5,
            note: 60,
            instrument: 0,
            channel: 0,
            velocity: 64,
            cluster_value: 0.0,
            beats: 0,
        }
    }

    #[test]
    fn valid_row_passes() {
        assert!(validate_row(&valid_row()).is_empty());
    }

    #[test]
    fn empty_label_fails() {
        let mut row = valid_row();
        row.label = String::new();
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn label_too_long_fails() {
        let mut row = valid_row();
        row.label = "a".repeat(256);
        let errs = validate_row(&row);
        assert!(errs.iter().any(|e| e.contains("255")));
    }

    #[test]
    fn nonfinite_x_fails() {
        let mut row = valid_row();
        row.x = f64::INFINITY;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn nan_y_fails() {
        let mut row = valid_row();
        row.y = f64::NAN;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn zero_size_fails() {
        let mut row = valid_row();
        row.size = 0.0;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn negative_size_fails() {
        let mut row = valid_row();
        row.size = -1.0;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn negative_size_alpha_fails() {
        let mut row = valid_row();
        row.size_alpha = -0.001;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn color_over_one_fails() {
        let mut row = valid_row();
        row.color_r = 1.001;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn color_negative_fails() {
        let mut row = valid_row();
        row.color_b = -0.1;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn velocity_zero_fails() {
        let mut row = valid_row();
        row.velocity = 0;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn velocity_128_fails() {
        let mut row = valid_row();
        row.velocity = 128;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn instrument_366_fails() {
        let mut row = valid_row();
        row.instrument = 366;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn instrument_365_passes() {
        let mut row = valid_row();
        row.instrument = 365;
        assert!(validate_row(&row).is_empty());
    }

    #[test]
    fn channel_16_fails() {
        let mut row = valid_row();
        row.channel = 16;
        assert!(!validate_row(&row).is_empty());
    }

    #[test]
    fn edge_unknown_label_fails() {
        use std::collections::HashSet;
        let edge = EdgeRow {
            from: "A".into(),
            to: "Z_UNKNOWN".into(),
            strength: 0.5,
        };
        let known: HashSet<&str> = ["A", "B"].iter().copied().collect();
        let errs = validate_edge(&edge, &known, "Sheet1");
        assert!(errs.iter().any(|e| e.contains("Z_UNKNOWN")));
    }

    #[test]
    fn edge_nonfinite_strength_fails() {
        use std::collections::HashSet;
        let edge = EdgeRow {
            from: "A".into(),
            to: "B".into(),
            strength: f64::NAN,
        };
        let known: HashSet<&str> = ["A", "B"].iter().copied().collect();
        let errs = validate_edge(&edge, &known, "Sheet1");
        assert!(!errs.is_empty());
    }

    #[test]
    fn dataset_validation_collects_all_errors() {
        let mut row = valid_row();
        row.label = String::new(); // invalid
        row.size = 0.0; // also invalid
        let sheet = LvSheet {
            name: "Sheet1".into(),
            sheet_index: 0,
            rows: vec![row],
            edges: vec![],
        };
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![sheet],
            all_labels: vec![],
        };
        let result = validate_dataset(&dataset);
        assert!(result.is_err());
        // Should report 3 failures (all_labels mismatch + empty label + zero size)
        if let Err(DataError::Validation { count, .. }) = result {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn note_128_fails() {
        let mut row = valid_row();
        row.note = 128;
        let errs = validate_row(&row);
        assert!(
            errs.iter().any(|e| e.contains("note")),
            "note=128 should fail validation"
        );
    }

    #[test]
    fn note_127_passes() {
        let mut row = valid_row();
        row.note = 127;
        assert!(
            validate_row(&row).is_empty(),
            "note=127 should pass validation"
        );
    }

    #[test]
    fn dataset_validation_rejects_stale_all_labels() {
        let dataset = LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "Sheet1".into(),
                sheet_index: 0,
                rows: vec![valid_row()],
                edges: vec![],
            }],
            all_labels: vec!["ghost".into(), "TestNode".into()],
        };

        let result = validate_dataset(&dataset);
        assert!(result.is_err());
        if let Err(DataError::Validation { messages, .. }) = result {
            assert!(messages.contains("all_labels does not match"));
        }
    }
}
