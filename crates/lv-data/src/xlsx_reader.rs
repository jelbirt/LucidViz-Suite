use std::io::{Read, Seek};
use std::path::Path;

use calamine::{open_workbook_auto, open_workbook_auto_from_rs, Data, DataType, Reader};

use crate::{validate_dataset, DataError, EdgeRow, LvDataset, LvRow, LvSheet, ShapeKind};

const MAX_XLSX_BYTES: u64 = 128 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Read an LV workbook from a file path.
pub fn read_lv_xlsx(path: &Path) -> Result<LvDataset, DataError> {
    let metadata = std::fs::metadata(path)?;
    if metadata.len() > MAX_XLSX_BYTES {
        return Err(DataError::FileTooLarge {
            path: path.display().to_string(),
            bytes: metadata.len(),
            limit: MAX_XLSX_BYTES,
        });
    }
    let mut workbook = open_workbook_auto(path)?;
    parse_workbook(&mut workbook, Some(path.to_path_buf()))
}

/// Read an LV workbook from an in-memory byte slice.
///
/// This variant is WASM-compatible (no filesystem access required).
pub fn read_lv_xlsx_bytes(bytes: &[u8]) -> Result<LvDataset, DataError> {
    if bytes.len() as u64 > MAX_XLSX_BYTES {
        return Err(DataError::FileTooLarge {
            path: "<memory>".into(),
            bytes: bytes.len() as u64,
            limit: MAX_XLSX_BYTES,
        });
    }
    let cursor = std::io::Cursor::new(bytes);
    let mut workbook = open_workbook_auto_from_rs(cursor)?;
    parse_workbook(&mut workbook, None)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal parsing
// ─────────────────────────────────────────────────────────────────────────────

fn parse_workbook<RS>(
    workbook: &mut calamine::Sheets<RS>,
    source_path: Option<std::path::PathBuf>,
) -> Result<LvDataset, DataError>
where
    RS: Read + Seek,
{
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        return Err(DataError::NoSheets);
    }

    let mut sheets: Vec<LvSheet> = Vec::new();

    for (sheet_index, name) in sheet_names.iter().enumerate() {
        let range = workbook.worksheet_range(name)?;
        let lv_sheet = parse_sheet(range, name, sheet_index)?;

        sheets.push(lv_sheet);
    }

    let all_labels = LvDataset::canonical_all_labels_from_sheets(&sheets);
    let dataset = LvDataset {
        source_path,
        sheets,
        all_labels,
    };
    validate_dataset(&dataset)?;
    Ok(dataset)
}

fn parse_sheet(
    range: calamine::Range<Data>,
    name: &str,
    sheet_index: usize,
) -> Result<LvSheet, DataError> {
    // Stream rows: scan for the edge boundary row-by-row without collecting
    // the entire sheet into memory. We need two passes: one to find the edge
    // section boundary, one to parse rows. Since calamine's Range is
    // cheaply iterable, this is efficient.
    let total_rows = range.rows().count();

    // Find edge section boundary (first row where col-0 is "from").
    let edge_start = range.rows().enumerate().position(|(_, row)| {
        row.first()
            .and_then(|c: &Data| c.get_string())
            .map(|s| s.trim().eq_ignore_ascii_case("from"))
            .unwrap_or(false)
    });

    let object_end = edge_start.unwrap_or(total_rows);

    // Parse object rows (skip header at row 0).
    let mut rows: Vec<LvRow> = Vec::new();
    for (i, raw) in range.rows().enumerate() {
        if i == 0 || i >= object_end {
            continue;
        }
        let raw_vec: Vec<Data> = raw.to_vec();
        if let Some(row) = parse_lv_row(&raw_vec, name, i + 1)? {
            rows.push(row);
        }
    }

    if rows.is_empty() {
        return Err(DataError::EmptySheet {
            sheet: name.to_owned(),
        });
    }

    // Parse edge rows (after the "from" header).
    let mut edges: Vec<EdgeRow> = Vec::new();
    if let Some(start) = edge_start {
        for (i, raw) in range.rows().enumerate() {
            if i <= start {
                continue;
            }
            let raw_vec: Vec<Data> = raw.to_vec();
            if let Some(edge) = parse_edge_row(&raw_vec, name, i + 1)? {
                edges.push(edge);
            }
        }
    }

    Ok(LvSheet {
        name: name.to_owned(),
        sheet_index,
        rows,
        edges,
    })
}

/// Returns `Ok(None)` for blank/sentinel rows.
fn parse_lv_row(raw: &[Data], sheet: &str, row_num: usize) -> Result<Option<LvRow>, DataError> {
    match raw.first() {
        None | Some(Data::Empty) => return Ok(None),
        _ => {}
    }

    let mk_err = |col: usize, expected: &'static str, actual: &Data| DataError::WrongCellType {
        sheet: sheet.to_owned(),
        row: row_num,
        col,
        expected,
        actual: format!("{actual:?}"),
    };

    let label = cell_string(raw, 0).ok_or_else(|| DataError::MalformedRow {
        sheet: sheet.to_owned(),
        row: row_num,
        msg: "column 0 (label) is missing or not a string".into(),
    })?;
    if label.is_empty() {
        return Ok(None);
    }

    let x = cell_f64(raw, 1).ok_or_else(|| mk_err(1, "number (x)", &cell_raw(raw, 1)))?;
    let y = cell_f64(raw, 2).ok_or_else(|| mk_err(2, "number (y)", &cell_raw(raw, 2)))?;
    let z = cell_f64(raw, 3).ok_or_else(|| mk_err(3, "number (z)", &cell_raw(raw, 3)))?;
    let size = cell_f64(raw, 4).ok_or_else(|| mk_err(4, "number (size)", &cell_raw(raw, 4)))?;
    let size_alpha =
        cell_f64(raw, 5).ok_or_else(|| mk_err(5, "number (size_alpha)", &cell_raw(raw, 5)))?;
    let spin_x = cell_f64(raw, 6).ok_or_else(|| mk_err(6, "number (spin_x)", &cell_raw(raw, 6)))?;
    let spin_y = cell_f64(raw, 7).ok_or_else(|| mk_err(7, "number (spin_y)", &cell_raw(raw, 7)))?;
    let spin_z = cell_f64(raw, 8).ok_or_else(|| mk_err(8, "number (spin_z)", &cell_raw(raw, 8)))?;

    let shape_str = cell_string(raw, 9).unwrap_or_else(|| "sphere".into());
    let shape: ShapeKind = shape_str.parse()?;

    let color_r =
        cell_f64(raw, 10).ok_or_else(|| mk_err(10, "number (color_r)", &cell_raw(raw, 10)))? as f32;
    let color_g =
        cell_f64(raw, 11).ok_or_else(|| mk_err(11, "number (color_g)", &cell_raw(raw, 11)))? as f32;
    let color_b =
        cell_f64(raw, 12).ok_or_else(|| mk_err(12, "number (color_b)", &cell_raw(raw, 12)))? as f32;

    let note = optional_integer_cell(raw, 13, sheet, row_num, 60.0, 0.0, 127.0)? as u8;
    let instrument = optional_integer_cell(raw, 14, sheet, row_num, 0.0, 0.0, 365.0)? as u16;
    let channel = optional_integer_cell(raw, 15, sheet, row_num, 0.0, 0.0, 15.0)? as u8;
    let velocity = optional_integer_cell(raw, 16, sheet, row_num, 64.0, 1.0, 127.0)? as u8;
    let cluster_value = optional_finite_cell(raw, 17, sheet, row_num, 0.0)?;
    let beats = optional_integer_cell(raw, 18, sheet, row_num, 0.0, 0.0, u32::MAX as f64)? as u32;

    Ok(Some(LvRow {
        label,
        x,
        y,
        z,
        size,
        size_alpha,
        spin_x,
        spin_y,
        spin_z,
        shape,
        color_r,
        color_g,
        color_b,
        note,
        instrument,
        channel,
        velocity,
        cluster_value,
        beats,
    }))
}

fn parse_edge_row(raw: &[Data], sheet: &str, row_num: usize) -> Result<Option<EdgeRow>, DataError> {
    let Some(from) = cell_string(raw, 0) else {
        return Ok(None);
    };
    let Some(to) = cell_string(raw, 1) else {
        return Ok(None);
    };
    if from.is_empty() || to.is_empty() {
        return Ok(None);
    }
    let strength = optional_finite_cell(raw, 2, sheet, row_num, 1.0)?;
    Ok(Some(EdgeRow { from, to, strength }))
}

fn optional_integer_cell(
    raw: &[Data],
    col: usize,
    sheet: &str,
    row_num: usize,
    default: f64,
    min: f64,
    max: f64,
) -> Result<f64, DataError> {
    let Some(value) = cell_optional_f64(raw, col, sheet, row_num)? else {
        return Ok(default);
    };
    if !value.is_finite() || value.fract() != 0.0 || value < min || value > max {
        return Err(DataError::ValueOutOfRange {
            sheet: sheet.to_owned(),
            row: row_num,
            col,
            msg: format!("expected integer in range [{min}, {max}], got {value}"),
        });
    }
    Ok(value)
}

fn optional_finite_cell(
    raw: &[Data],
    col: usize,
    sheet: &str,
    row_num: usize,
    default: f64,
) -> Result<f64, DataError> {
    let Some(value) = cell_optional_f64(raw, col, sheet, row_num)? else {
        return Ok(default);
    };
    if !value.is_finite() {
        return Err(DataError::ValueOutOfRange {
            sheet: sheet.to_owned(),
            row: row_num,
            col,
            msg: format!("expected finite numeric value, got {value}"),
        });
    }
    Ok(value)
}

fn cell_optional_f64(
    raw: &[Data],
    col: usize,
    sheet: &str,
    row_num: usize,
) -> Result<Option<f64>, DataError> {
    match raw.get(col) {
        None | Some(Data::Empty) => Ok(None),
        Some(Data::Float(v)) => Ok(Some(*v)),
        Some(Data::Int(v)) => Ok(Some(*v as f64)),
        Some(Data::String(s)) => {
            s.trim()
                .parse::<f64>()
                .map(Some)
                .map_err(|_| DataError::WrongCellType {
                    sheet: sheet.to_owned(),
                    row: row_num,
                    col,
                    expected: "numeric value",
                    actual: raw
                        .get(col)
                        .map(|v| format!("{v:?}"))
                        .unwrap_or_else(|| "missing".to_string()),
                })
        }
        Some(other) => Err(DataError::WrongCellType {
            sheet: sheet.to_owned(),
            row: row_num,
            col,
            expected: "numeric value",
            actual: format!("{other:?}"),
        }),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell helpers
// ─────────────────────────────────────────────────────────────────────────────

fn cell_raw(raw: &[Data], col: usize) -> Data {
    raw.get(col).cloned().unwrap_or(Data::Empty)
}

fn cell_f64(raw: &[Data], col: usize) -> Option<f64> {
    match raw.get(col)? {
        Data::Float(v) => Some(*v),
        Data::Int(v) => Some(*v as f64),
        Data::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn cell_string(raw: &[Data], col: usize) -> Option<String> {
    match raw.get(col)? {
        Data::String(s) => Some(s.trim().to_owned()),
        Data::Float(v) => Some(v.to_string()),
        Data::Int(v) => Some(v.to_string()),
        Data::Bool(b) => Some(b.to_string()),
        Data::Empty => None,
        other => Some(format!("{other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_row() -> Vec<Data> {
        vec![
            Data::String("node".into()),
            Data::Float(0.0),
            Data::Float(0.0),
            Data::Float(0.0),
            Data::Float(1.0),
            Data::Float(0.5),
            Data::Float(0.0),
            Data::Float(0.0),
            Data::Float(0.0),
            Data::String("sphere".into()),
            Data::Float(1.0),
            Data::Float(1.0),
            Data::Float(1.0),
            Data::Float(60.0),
            Data::Float(0.0),
            Data::Float(0.0),
            Data::Float(64.0),
            Data::Float(0.0),
            Data::Float(1.0),
        ]
    }

    #[test]
    fn parse_lv_row_rejects_fractional_velocity() {
        let mut row = base_row();
        row[16] = Data::Float(64.5);
        let err = parse_lv_row(&row, "Sheet1", 2).expect_err("fractional velocity should fail");
        assert!(matches!(err, DataError::ValueOutOfRange { col: 16, .. }));
    }

    #[test]
    fn parse_edge_row_rejects_non_finite_strength() {
        let row = vec![
            Data::String("a".into()),
            Data::String("b".into()),
            Data::String("NaN".into()),
        ];
        let err = parse_edge_row(&row, "Sheet1", 4).expect_err("NaN strength should fail");
        assert!(matches!(err, DataError::ValueOutOfRange { col: 2, .. }));
    }

    #[test]
    fn read_lv_xlsx_bytes_rejects_oversized_input() {
        let bytes = vec![0; (MAX_XLSX_BYTES + 1) as usize];
        let err = read_lv_xlsx_bytes(&bytes).expect_err("oversized xlsx should fail");
        assert!(matches!(err, DataError::FileTooLarge { .. }));
    }
}
