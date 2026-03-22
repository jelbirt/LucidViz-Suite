//! Output: write MF pipeline results to JSON and XLSX.

use std::path::Path;

use anyhow::Result;
use rust_xlsxwriter::{Format, Workbook};

use crate::error::MfError;
use crate::types::{MfOutput, MfSeriesOutput};

/// Write the MF pipeline output as JSON.
pub fn write_mf_json(output: &MfOutput, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    atomic_write(path, json.as_bytes())?;
    Ok(())
}

/// Write the MF temporal series output as JSON.
pub fn write_mf_series_json(output: &MfSeriesOutput, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    atomic_write(path, json.as_bytes())?;
    Ok(())
}

/// Write the MF pipeline output as an XLSX workbook with 5 sheets.
///
/// Sheets:
///  1. "Vocabulary" — token, degree, distance, closeness, betweenness
///  2. "NPPMI Matrix" — normalized positive PMI values
///  3. "PPMI Matrix" — unnormalized positive PMI values
///  4. "Similarity Matrix" — runtime-selected similarity matrix
///  5. "Raw Counts" — raw co-occurrence counts
pub fn write_mf_xlsx(output: &MfOutput, path: &Path, raw_counts: &[u64]) -> Result<()> {
    let mut wb = Workbook::new();
    let bold = Format::new().set_bold();
    let n = output.n;

    // ---- Sheet 1: Vocabulary ----
    {
        let ws = wb.add_worksheet();
        ws.set_name("Vocabulary")
            .map_err(|e| MfError::Xlsx(e.to_string()))?;

        let headers = ["Token", "Degree", "Distance", "Closeness", "Betweenness"];
        for (c, h) in headers.iter().enumerate() {
            ws.write_with_format(0, c as u16, *h, &bold)
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
        }
        for (i, label) in output.labels.iter().enumerate() {
            let row = (i + 1) as u32;
            ws.write(row, 0, label.as_str())
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
            ws.write(row, 1, output.centrality.degree[i])
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
            ws.write(row, 2, output.centrality.distance[i])
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
            ws.write(row, 3, output.centrality.closeness[i])
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
            ws.write(row, 4, output.centrality.betweenness[i])
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
        }
    }

    // ---- Sheet 2: NPPMI Matrix ----
    write_square_matrix_sheet(
        &mut wb,
        "NPPMI Matrix",
        &output.labels,
        &output.nppmi_matrix,
        n,
        &bold,
    )?;

    // ---- Sheet 3: PPMI Matrix ----
    write_square_matrix_sheet(
        &mut wb,
        "PPMI Matrix",
        &output.labels,
        &output.ppmi_matrix,
        n,
        &bold,
    )?;

    // ---- Sheet 4: Similarity Matrix ----
    write_square_matrix_sheet(
        &mut wb,
        "Similarity Matrix",
        &output.labels,
        &output.similarity_matrix,
        n,
        &bold,
    )?;

    // ---- Sheet 5: Raw Counts ----
    {
        let ws = wb.add_worksheet();
        ws.set_name("Raw Counts")
            .map_err(|e| MfError::Xlsx(e.to_string()))?;

        // Header.
        ws.write_with_format(0, 0, "Token", &bold)
            .map_err(|e| MfError::Xlsx(e.to_string()))?;
        for (j, label) in output.labels.iter().enumerate() {
            ws.write_with_format(0, (j + 1) as u16, label.as_str(), &bold)
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
        }
        for (i, label) in output.labels.iter().enumerate() {
            let row = (i + 1) as u32;
            ws.write(row, 0, label.as_str())
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
            for j in 0..n {
                let count = if i < raw_counts.len() / n && j < n {
                    raw_counts[i * n + j] as f64
                } else {
                    0.0
                };
                ws.write(row, (j + 1) as u16, count)
                    .map_err(|e| MfError::Xlsx(e.to_string()))?;
            }
        }
    }

    save_workbook_atomic(&mut wb, path)?;
    Ok(())
}

fn write_square_matrix_sheet(
    wb: &mut Workbook,
    name: &str,
    labels: &[String],
    data: &[f64],
    n: usize,
    bold: &Format,
) -> Result<()> {
    let ws = wb.add_worksheet();
    ws.set_name(name)
        .map_err(|e| MfError::Xlsx(e.to_string()))?;

    ws.write_with_format(0, 0, "Token", bold)
        .map_err(|e| MfError::Xlsx(e.to_string()))?;
    for (j, label) in labels.iter().enumerate() {
        ws.write_with_format(0, (j + 1) as u16, label.as_str(), bold)
            .map_err(|e| MfError::Xlsx(e.to_string()))?;
    }
    for (i, label) in labels.iter().enumerate() {
        let row = (i + 1) as u32;
        ws.write(row, 0, label.as_str())
            .map_err(|e| MfError::Xlsx(e.to_string()))?;
        for j in 0..n {
            ws.write(row, (j + 1) as u16, data[i * n + j])
                .map_err(|e| MfError::Xlsx(e.to_string()))?;
        }
    }

    Ok(())
}

fn save_workbook_atomic(wb: &mut Workbook, path: &Path) -> Result<()> {
    let tmp_path = temp_path(path)?;
    wb.save(&tmp_path)
        .map_err(|e| MfError::Xlsx(e.to_string()))?;
    replace_file(&tmp_path, path)?;
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp_path = temp_path(path)?;
    std::fs::write(&tmp_path, bytes)?;
    replace_file(&tmp_path, path)?;
    Ok(())
}

fn temp_path(path: &Path) -> Result<std::path::PathBuf> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            MfError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path must have file name",
            ))
        })?;
    Ok(path.with_file_name(format!(".{file_name}.tmp-{}", std::process::id())))
}

fn replace_file(tmp_path: &Path, path: &Path) -> Result<()> {
    if let Err(err) = std::fs::rename(tmp_path, path) {
        if path.exists() {
            let _ = std::fs::remove_file(path);
            std::fs::rename(tmp_path, path)?;
        } else {
            return Err(err.into());
        }
    }
    Ok(())
}
