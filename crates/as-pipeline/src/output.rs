//! XLSX and JSON output for AlignSpace pipeline results.

use std::path::Path;

use anyhow::{Context, Result};
use rust_xlsxwriter::{Format, Workbook};

use crate::types::{AsPipelineResult, CentralityReport, MdsCoordinates, SeMatrix};

/// Write the full AS pipeline result to a directory.
///
/// Produces:
///   - `ET-V.xlsx`                   (ETV dataset)
///   - `Coordinates.xlsx`            (MDS coordinates for each time step)
///   - `Centralities.xlsx`           (Centrality report for each time step)
///   - `as_result.json`              (full result as JSON)
pub fn write_as_results(result: &AsPipelineResult, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;

    // Write ETV dataset via lv-data.
    let etv_path = out_dir.join("ET-V.xlsx");
    lv_data::xlsx_writer::write_etv_xlsx(&result.etv_dataset, &etv_path)
        .context("Failed to write ET-V.xlsx")?;

    // Write coordinates workbook.
    write_coordinates_xlsx(&result.coordinates, &out_dir.join("Coordinates.xlsx"))?;

    // Write centralities workbook.
    write_centralities_xlsx(&result.centralities, &out_dir.join("Centralities.xlsx"))?;

    // Write JSON.
    write_as_json(result, &out_dir.join("as_result.json"))?;

    Ok(())
}

fn write_coordinates_xlsx(coords: &[MdsCoordinates], path: &Path) -> Result<()> {
    let mut wb = Workbook::new();
    let bold = Format::new().set_bold();

    for (step_idx, mds) in coords.iter().enumerate() {
        let sheet_name = format!("Step_{}", step_idx + 1);
        let ws = wb.add_worksheet();
        ws.set_name(&sheet_name)
            .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;

        // Header row: Label, Dim1, Dim2, ...
        ws.write_with_format(0, 0, "Label", &bold)
            .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
        for d in 0..mds.dims {
            ws.write_with_format(0, (d + 1) as u16, format!("Dim{}", d + 1), &bold)
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
        }

        // Data rows.
        for (i, label) in mds.labels.iter().enumerate() {
            let row = (i + 1) as u32;
            ws.write(row, 0, label.as_str())
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            for d in 0..mds.dims {
                ws.write(row, (d + 1) as u16, mds.data[i * mds.dims + d])
                    .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            }
        }
    }

    wb.save(path)
        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
    Ok(())
}

fn write_centralities_xlsx(reports: &[CentralityReport], path: &Path) -> Result<()> {
    let mut wb = Workbook::new();
    let bold = Format::new().set_bold();

    for (step_idx, report) in reports.iter().enumerate() {
        let sheet_name = format!("Step_{}", step_idx + 1);
        let ws = wb.add_worksheet();
        ws.set_name(&sheet_name)
            .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;

        // Header.
        let headers = ["Label", "Degree", "Distance", "Closeness", "Betweenness"];
        for (c, h) in headers.iter().enumerate() {
            ws.write_with_format(0, c as u16, *h, &bold)
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
        }

        for (i, label) in report.labels.iter().enumerate() {
            let row = (i + 1) as u32;
            ws.write(row, 0, label.as_str())
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            ws.write(row, 1, report.degree[i])
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            ws.write(row, 2, report.distance[i])
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            ws.write(row, 3, report.closeness[i])
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            ws.write(row, 4, report.betweenness[i])
                .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
        }
    }

    wb.save(path)
        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
    Ok(())
}

fn write_as_json(result: &AsPipelineResult, path: &Path) -> Result<()> {
    #[derive(serde::Serialize)]
    struct JsonResult<'a> {
        coordinates: &'a [MdsCoordinates],
        centralities: &'a [CentralityReport],
        se_matrices: &'a [SeMatrix],
    }

    let jr = JsonResult {
        coordinates: &result.coordinates,
        centralities: &result.centralities,
        se_matrices: &result.se_matrices,
    };

    let json = serde_json::to_string_pretty(&jr)?;
    std::fs::write(path, json)?;
    Ok(())
}
