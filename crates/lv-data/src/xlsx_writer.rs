use std::path::Path;

use rust_xlsxwriter::{Format, Workbook};

use crate::{DataError, LvDataset, LvRow, LvSheet};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Write an [`LvDataset`] to an XLSX file at `path`.
pub fn write_lv_xlsx(dataset: &LvDataset, path: &Path) -> Result<(), DataError> {
    let bytes = write_lv_xlsx_bytes(dataset)?;
    crate::io_util::atomic_write(path, &bytes)?;
    Ok(())
}

/// Serialise an [`LvDataset`] to an in-memory XLSX byte vector.
///
/// This variant is WASM-compatible (no filesystem access required).
pub fn write_lv_xlsx_bytes(dataset: &LvDataset) -> Result<Vec<u8>, DataError> {
    let mut workbook = Workbook::new();

    let header_fmt = Format::new().set_bold();

    for lv_sheet in &dataset.sheets {
        let ws = workbook.add_worksheet();
        ws.set_name(&lv_sheet.name).map_err(DataError::XlsxWrite)?;

        write_sheet(ws, lv_sheet, &header_fmt)?;
    }

    let bytes = workbook.save_to_buffer().map_err(DataError::XlsxWrite)?;
    Ok(bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal
// ─────────────────────────────────────────────────────────────────────────────

const HEADERS: &[&str] = &[
    "label",
    "x",
    "y",
    "z",
    "size",
    "size_alpha",
    "spin_x",
    "spin_y",
    "spin_z",
    "shape",
    "color_r",
    "color_g",
    "color_b",
    "note",
    "instrument",
    "channel",
    "velocity",
    "cluster_value",
    "beats",
];

const EDGE_HEADERS: &[&str] = &["from", "to", "strength"];

fn write_sheet(
    ws: &mut rust_xlsxwriter::Worksheet,
    sheet: &LvSheet,
    header_fmt: &Format,
) -> Result<(), DataError> {
    // ── Header row ───────────────────────────────────────────────────────────
    for (col, &h) in HEADERS.iter().enumerate() {
        ws.write_with_format(0, col as u16, h, header_fmt)
            .map_err(DataError::XlsxWrite)?;
    }

    // ── Object rows ──────────────────────────────────────────────────────────
    for (i, row) in sheet.rows.iter().enumerate() {
        write_lv_row(ws, i as u32 + 1, row)?;
    }

    // ── Edge section ─────────────────────────────────────────────────────────
    if !sheet.edges.is_empty() {
        let edge_header_row = sheet.rows.len() as u32 + 2; // blank separator row

        for (col, &h) in EDGE_HEADERS.iter().enumerate() {
            ws.write_with_format(edge_header_row, col as u16, h, header_fmt)
                .map_err(DataError::XlsxWrite)?;
        }

        for (i, edge) in sheet.edges.iter().enumerate() {
            let row = edge_header_row + 1 + i as u32;
            ws.write(row, 0, edge.from.as_str())
                .map_err(DataError::XlsxWrite)?;
            ws.write(row, 1, edge.to.as_str())
                .map_err(DataError::XlsxWrite)?;
            ws.write(row, 2, edge.strength)
                .map_err(DataError::XlsxWrite)?;
        }
    }

    Ok(())
}

fn write_lv_row(
    ws: &mut rust_xlsxwriter::Worksheet,
    row: u32,
    lv: &LvRow,
) -> Result<(), DataError> {
    macro_rules! w {
        ($col:expr, $val:expr) => {
            ws.write(row, $col, $val).map_err(DataError::XlsxWrite)?
        };
    }

    w!(0, lv.label.as_str());
    w!(1, lv.x);
    w!(2, lv.y);
    w!(3, lv.z);
    w!(4, lv.size);
    w!(5, lv.size_alpha);
    w!(6, lv.spin_x);
    w!(7, lv.spin_y);
    w!(8, lv.spin_z);
    w!(9, lv.shape.to_string().as_str());
    w!(10, lv.color_r as f64);
    w!(11, lv.color_g as f64);
    w!(12, lv.color_b as f64);
    w!(13, u32::from(lv.note));
    w!(14, u32::from(lv.instrument));
    w!(15, u32::from(lv.channel));
    w!(16, u32::from(lv.velocity));
    w!(17, lv.cluster_value);
    w!(18, lv.beats);

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EdgeRow, LvRow, LvSheet, ShapeKind};

    fn sample_dataset() -> LvDataset {
        let rows = vec![
            LvRow {
                label: "Alpha".into(),
                x: 1.0,
                y: 2.0,
                z: 3.0,
                size: 1.5,
                size_alpha: 0.25,
                spin_x: 0.0,
                spin_y: 45.0,
                spin_z: 0.0,
                shape: ShapeKind::Cube,
                color_r: 0.8,
                color_g: 0.2,
                color_b: 0.5,
                note: 60,
                instrument: 1,
                channel: 0,
                velocity: 80,
                cluster_value: 1.0,
                beats: 2,
            },
            LvRow {
                label: "Beta".into(),
                x: -1.0,
                y: 0.0,
                z: 1.0,
                size: 0.8,
                size_alpha: 0.0,
                spin_x: 0.0,
                spin_y: 0.0,
                spin_z: 0.0,
                shape: ShapeKind::Sphere,
                color_r: 0.0,
                color_g: 1.0,
                color_b: 0.0,
                note: 48,
                instrument: 0,
                channel: 1,
                velocity: 64,
                cluster_value: 0.0,
                beats: 0,
            },
        ];
        let edges = vec![EdgeRow {
            from: "Alpha".into(),
            to: "Beta".into(),
            strength: 0.75,
        }];
        LvDataset {
            source_path: None,
            sheets: vec![LvSheet {
                name: "Sheet1".into(),
                sheet_index: 0,
                rows,
                edges,
            }],
            all_labels: vec!["Alpha".into(), "Beta".into()],
        }
    }

    #[test]
    fn write_to_bytes_succeeds() {
        let ds = sample_dataset();
        let bytes = write_lv_xlsx_bytes(&ds).expect("write should succeed");
        assert!(!bytes.is_empty());
        // XLSX files start with PK (ZIP magic bytes)
        assert_eq!(&bytes[..2], b"PK");
    }

    #[test]
    fn write_then_read_roundtrip() {
        use crate::xlsx_reader::read_lv_xlsx_bytes;

        let original = sample_dataset();
        let bytes = write_lv_xlsx_bytes(&original).expect("write");
        let recovered = read_lv_xlsx_bytes(&bytes).expect("read");

        assert_eq!(recovered.sheets.len(), 1);
        let sheet = &recovered.sheets[0];
        assert_eq!(sheet.rows.len(), 2);

        let a = &sheet.rows[0];
        assert_eq!(a.label, "Alpha");
        assert_eq!(a.shape, ShapeKind::Cube);
        assert!((a.x - 1.0).abs() < 1e-9);
        assert!((a.size - 1.5).abs() < 1e-9);
        assert_eq!(a.velocity, 80);
        assert_eq!(a.beats, 2);

        let b = &sheet.rows[1];
        assert_eq!(b.label, "Beta");
        assert_eq!(b.shape, ShapeKind::Sphere);

        // Edges
        assert_eq!(sheet.edges.len(), 1);
        assert_eq!(sheet.edges[0].from, "Alpha");
        assert_eq!(sheet.edges[0].to, "Beta");
        assert!((sheet.edges[0].strength - 0.75).abs() < 1e-9);
    }
}
