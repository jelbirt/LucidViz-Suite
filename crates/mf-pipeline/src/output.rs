//! Output: write MF pipeline results to JSON and XLSX.

use std::path::Path;

use anyhow::Result;
use rust_xlsxwriter::{Format, Workbook};

use crate::error::MfError;
use crate::types::{MfOutput, MfSeriesOutput};

/// Write the MF pipeline output as JSON.
pub fn write_mf_json(output: &MfOutput, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write the MF temporal series output as JSON.
pub fn write_mf_series_json(output: &MfSeriesOutput, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(output)?;
    std::fs::write(path, json)?;
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

    wb.save(path).map_err(|e| MfError::Xlsx(e.to_string()))?;
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
