use std::io::{Read, Seek};
use std::path::Path;

use calamine::{open_workbook_auto, open_workbook_auto_from_rs, Data, DataType, Reader};

use crate::{DataError, EdgeRow, EtvDataset, EtvRow, EtvSheet, ShapeKind};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Read an ETV workbook from a file path.
pub fn read_etv_xlsx(path: &Path) -> Result<EtvDataset, DataError> {
    let mut workbook = open_workbook_auto(path)?;
    parse_workbook(&mut workbook, Some(path.to_path_buf()))
}

/// Read an ETV workbook from an in-memory byte slice.
///
/// This variant is WASM-compatible (no filesystem access required).
pub fn read_etv_xlsx_bytes(bytes: &[u8]) -> Result<EtvDataset, DataError> {
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
) -> Result<EtvDataset, DataError>
where
    RS: Read + Seek,
{
    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        return Err(DataError::NoSheets);
    }

    let mut sheets: Vec<EtvSheet> = Vec::new();
    let mut all_labels: Vec<String> = Vec::new();
    let mut seen_labels: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (sheet_index, name) in sheet_names.iter().enumerate() {
        let range = workbook.worksheet_range(name)?;
        let etv_sheet = parse_sheet(range, name, sheet_index)?;

        for row in &etv_sheet.rows {
            if seen_labels.insert(row.label.clone()) {
                all_labels.push(row.label.clone());
            }
        }

        sheets.push(etv_sheet);
    }

    Ok(EtvDataset {
        source_path,
        sheets,
        all_labels,
    })
}

fn parse_sheet(
    range: calamine::Range<Data>,
    name: &str,
    sheet_index: usize,
) -> Result<EtvSheet, DataError> {
    // Collect all rows up-front so we can search for the edge boundary
    let rows_raw: Vec<Vec<Data>> = range.rows().map(|r| r.to_vec()).collect();

    // Edge section starts at the first row where col-0 is "from" (case-insensitive)
    let edge_start = rows_raw.iter().position(|row| {
        row.first()
            .and_then(|c: &Data| c.get_string())
            .map(|s| s.trim().eq_ignore_ascii_case("from"))
            .unwrap_or(false)
    });

    // Row 0 is the column-header row — skip it.  Object rows end before edge section.
    let object_end = edge_start.unwrap_or(rows_raw.len());
    let object_rows = if rows_raw.len() <= 1 {
        &rows_raw[rows_raw.len()..] // empty slice
    } else {
        &rows_raw[1..object_end]
    };

    let mut rows: Vec<EtvRow> = Vec::with_capacity(object_rows.len());
    for (i, raw) in object_rows.iter().enumerate() {
        // i+2 because row 0 = header, row 1 = first data row (1-based)
        if let Some(row) = parse_etv_row(raw, name, i + 2)? {
            rows.push(row);
        }
    }

    if rows.is_empty() {
        return Err(DataError::EmptySheet {
            sheet: name.to_owned(),
        });
    }

    let mut edges: Vec<EdgeRow> = Vec::new();
    if let Some(start) = edge_start {
        for raw in rows_raw.iter().skip(start + 1) {
            if let Some(edge) = parse_edge_row(raw) {
                edges.push(edge);
            }
        }
    }

    Ok(EtvSheet {
        name: name.to_owned(),
        sheet_index,
        rows,
        edges,
    })
}

/// Returns `Ok(None)` for blank/sentinel rows.
fn parse_etv_row(raw: &[Data], sheet: &str, row_num: usize) -> Result<Option<EtvRow>, DataError> {
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

    let note = cell_f64(raw, 13).unwrap_or(60.0) as u8;
    let instrument = cell_f64(raw, 14).unwrap_or(0.0) as u16;
    let channel = cell_f64(raw, 15).unwrap_or(0.0) as u8;
    let velocity = cell_f64(raw, 16).unwrap_or(64.0) as u8;
    let cluster_value = cell_f64(raw, 17).unwrap_or(0.0);
    let beats = cell_f64(raw, 18).unwrap_or(0.0) as u32;

    Ok(Some(EtvRow {
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

fn parse_edge_row(raw: &[Data]) -> Option<EdgeRow> {
    let from = cell_string(raw, 0)?;
    let to = cell_string(raw, 1)?;
    let strength = cell_f64(raw, 2).unwrap_or(1.0);
    if from.is_empty() || to.is_empty() {
        return None;
    }
    Some(EdgeRow { from, to, strength })
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
