//! Output: write MF pipeline results to JSON and XLSX.

use std::path::Path;

use anyhow::Result;
use rust_xlsxwriter::{Format, Workbook};

use crate::error::MfError;
use crate::types::{MfOutput, MfSeriesOutput};

const MAX_MF_JSON_BYTES: u64 = 128 * 1024 * 1024;

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

/// Read the MF pipeline output from JSON with a bounded file-size check.
pub fn read_mf_json(path: &Path) -> Result<MfOutput> {
    let bytes = read_bounded_file(path, MAX_MF_JSON_BYTES)?;
    let output: MfOutput = serde_json::from_slice(&bytes)?;
    output.validate()?;
    Ok(output)
}

/// Read the MF temporal series output from JSON with a bounded file-size check.
pub fn read_mf_series_json(path: &Path) -> Result<MfSeriesOutput> {
    let bytes = read_bounded_file(path, MAX_MF_JSON_BYTES)?;
    let output: MfSeriesOutput = serde_json::from_slice(&bytes)?;
    output.validate()?;
    Ok(output)
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
            if output.centrality.distance[i].is_nan() {
                ws.write(row, 2, "N/A")
                    .map_err(|e| MfError::Xlsx(e.to_string()))?;
            } else {
                ws.write(row, 2, output.centrality.distance[i])
                    .map_err(|e| MfError::Xlsx(e.to_string()))?;
            }
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
    lv_data::io_util::save_workbook_atomic(wb, path).map_err(|e| MfError::Xlsx(e.to_string()))?;
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    lv_data::io_util::atomic_write(path, bytes)?;
    Ok(())
}

fn read_bounded_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    Ok(lv_data::io_util::read_bounded_file(path, limit)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SimToDistMethod;
    use as_pipeline::types::CentralityReport;

    fn sample_output() -> MfOutput {
        MfOutput {
            labels: vec!["alpha".into(), "beta".into()],
            similarity_matrix: vec![1.0, 0.5, 0.5, 1.0],
            sim_to_dist: SimToDistMethod::Linear,
            nppmi_matrix: vec![1.0, 0.5, 0.5, 1.0],
            raw_counts: vec![1, 2, 2, 1],
            ppmi_matrix: vec![1.0, 0.5, 0.5, 1.0],
            n: 2,
            centrality: CentralityReport {
                labels: vec!["alpha".into(), "beta".into()],
                degree: vec![1.0, 1.0],
                distance: vec![1.0, 1.0],
                closeness: vec![1.0, 1.0],
                betweenness: vec![0.0, 0.0],
                harmonic: vec![1.0, 1.0],
                eigenvector: vec![1.0, 1.0],
                pagerank: vec![1.0, 1.0],
            },
        }
    }

    #[test]
    fn mf_json_roundtrip_file_is_bounded_and_validated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mf_output.json");
        let output = sample_output();

        write_mf_json(&output, &path).expect("write json");
        let loaded = read_mf_json(&path).expect("read json");

        assert_eq!(loaded.labels, output.labels);
        assert_eq!(loaded.similarity_matrix, output.similarity_matrix);
    }

    #[test]
    fn mf_json_reader_rejects_oversized_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("oversized.json");
        std::fs::write(&path, vec![b'x'; 1024]).expect("write test file");

        let err = read_bounded_file(&path, 16).expect_err("expected oversized file failure");
        assert!(err.to_string().contains("exceeding limit"));
    }
}
