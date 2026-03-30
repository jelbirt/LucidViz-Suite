//! XLSX and JSON output for AlignSpace pipeline results.

use std::path::Path;

use anyhow::{Context, Result};
use rust_xlsxwriter::{Format, Workbook};

use crate::types::{
    AsPipelineResult, CentralityMode, CentralityState, DistanceMatrix, MdsCoordinates,
    ProcrustesResult,
};

/// Write the full AS pipeline result to a directory.
///
/// Produces:
///   - `LV.xlsx`                   (LV dataset)
///   - `Coordinates.xlsx`            (MDS coordinates for each time step)
///   - `Centralities.xlsx`           (Centrality report for each time step)
///   - `as_result.json`              (full result as JSON)
pub fn write_as_results(result: &AsPipelineResult, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;

    // Write LV dataset via lv-data.
    let lv_path = out_dir.join("LV.xlsx");
    lv_data::xlsx_writer::write_lv_xlsx(&result.lv_dataset, &lv_path)
        .context("Failed to write LV.xlsx")?;

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

    save_workbook_atomic(&mut wb, path)?;
    Ok(())
}

fn write_centralities_xlsx(reports: &[CentralityState], path: &Path) -> Result<()> {
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

        match report {
            CentralityState::Computed(report) => {
                for (i, label) in report.labels.iter().enumerate() {
                    let row = (i + 1) as u32;
                    ws.write(row, 0, label.as_str())
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                    ws.write(row, 1, report.degree[i])
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                    if report.distance[i].is_nan() {
                        ws.write(row, 2, "N/A")
                            .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                    } else {
                        ws.write(row, 2, report.distance[i])
                            .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                    }
                    ws.write(row, 3, report.closeness[i])
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                    ws.write(row, 4, report.betweenness[i])
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                }
            }
            CentralityState::Unavailable { labels, reason } => {
                for (i, label) in labels.iter().enumerate() {
                    let row = (i + 1) as u32;
                    ws.write(row, 0, label.as_str())
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                }
                ws.write(0, 5, "Status")
                    .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
                ws.write(1, 5, reason.as_str())
                    .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
            }
        }
    }

    save_workbook_atomic(&mut wb, path)?;
    Ok(())
}

fn write_as_json(result: &AsPipelineResult, path: &Path) -> Result<()> {
    #[derive(serde::Serialize)]
    struct JsonResult<'a> {
        coordinates: &'a [MdsCoordinates],
        procrustes: &'a [ProcrustesResult],
        centralities: &'a [CentralityState],
        centrality_mode: CentralityMode,
        distance_matrices: &'a [DistanceMatrix],
        lv_dataset: &'a lv_data::schema::LvDataset,
    }

    let jr = JsonResult {
        coordinates: &result.coordinates,
        procrustes: &result.procrustes,
        centralities: &result.centralities,
        centrality_mode: result.centrality_mode,
        distance_matrices: &result.distance_matrices,
        lv_dataset: &result.lv_dataset,
    };

    let json = serde_json::to_string_pretty(&jr)?;
    atomic_write(path, json.as_bytes())?;
    Ok(())
}

fn save_workbook_atomic(wb: &mut Workbook, path: &Path) -> Result<()> {
    lv_data::io_util::save_workbook_atomic(wb, path)
        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    lv_data::io_util::atomic_write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_as_json;
    use crate::types::{
        AsPipelineResult, CentralityMode, CentralityState, DistanceMatrix, MdsAlgorithm,
        MdsCoordinates, ProcrustesResult,
    };
    use lv_data::schema::LvDataset;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn write_as_json_includes_procrustes_and_lv_dataset() {
        let result = AsPipelineResult {
            coordinates: vec![MdsCoordinates::new(
                vec!["alpha".into()],
                vec![1.0, 2.0],
                2,
                0.0,
                MdsAlgorithm::Classical,
            )
            .expect("test coordinates should build")],
            procrustes: vec![ProcrustesResult {
                aligned: MdsCoordinates::new(
                    vec!["alpha".into()],
                    vec![1.0, 2.0],
                    2,
                    0.0,
                    MdsAlgorithm::Classical,
                )
                .expect("test coordinates should build"),
                rotation: vec![1.0, 0.0, 0.0, 1.0],
                scale: 1.0,
                translation: vec![0.0, 0.0],
                residual: 0.0,
            }],
            centralities: vec![CentralityState::Unavailable {
                labels: vec!["alpha".into()],
                reason: "none".into(),
            }],
            centrality_mode: CentralityMode::Directed,
            distance_matrices: vec![DistanceMatrix::new(vec!["alpha".into()], vec![0.0])
                .expect("test distance matrix should build")],
            lv_dataset: LvDataset {
                source_path: None,
                sheets: vec![],
                all_labels: vec!["alpha".into()],
            },
        };

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time failed")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("as-output-{stamp}.json"));
        write_as_json(&result, &path).expect("json write failed");

        let json = std::fs::read_to_string(&path).expect("json read failed");
        assert!(json.contains("\"procrustes\""));
        assert!(json.contains("\"lv_dataset\""));
        assert!(json.contains("\"centrality_mode\""));

        let _ = std::fs::remove_file(path);
    }
}
