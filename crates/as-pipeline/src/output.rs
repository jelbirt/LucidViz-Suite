//! XLSX and JSON output for AlignSpace pipeline results.

use std::path::Path;

use anyhow::{Context, Result};
use rust_xlsxwriter::{Format, Workbook};

use crate::types::{
    AsPipelineResult, CentralityState, DistanceMatrix, MdsCoordinates, ProcrustesResult,
};

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
                    ws.write(row, 2, report.distance[i])
                        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
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

    wb.save(path)
        .map_err(|e| crate::error::AsError::Xlsx(e.to_string()))?;
    Ok(())
}

fn write_as_json(result: &AsPipelineResult, path: &Path) -> Result<()> {
    #[derive(serde::Serialize)]
    struct JsonResult<'a> {
        coordinates: &'a [MdsCoordinates],
        procrustes: &'a [ProcrustesResult],
        centralities: &'a [CentralityState],
        distance_matrices: &'a [DistanceMatrix],
        etv_dataset: &'a lv_data::schema::EtvDataset,
    }

    let jr = JsonResult {
        coordinates: &result.coordinates,
        procrustes: &result.procrustes,
        centralities: &result.centralities,
        distance_matrices: &result.distance_matrices,
        etv_dataset: &result.etv_dataset,
    };

    let json = serde_json::to_string_pretty(&jr)?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::write_as_json;
    use crate::types::{
        AsPipelineResult, CentralityState, DistanceMatrix, MdsAlgorithm, MdsCoordinates,
        ProcrustesResult,
    };
    use lv_data::schema::EtvDataset;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn write_as_json_includes_procrustes_and_etv_dataset() {
        let result = AsPipelineResult {
            coordinates: vec![MdsCoordinates::new(
                vec!["alpha".into()],
                vec![1.0, 2.0],
                2,
                0.0,
                MdsAlgorithm::Classical,
            )],
            procrustes: vec![ProcrustesResult {
                aligned: MdsCoordinates::new(
                    vec!["alpha".into()],
                    vec![1.0, 2.0],
                    2,
                    0.0,
                    MdsAlgorithm::Classical,
                ),
                rotation: vec![1.0, 0.0, 0.0, 1.0],
                scale: 1.0,
                translation: vec![0.0, 0.0],
                residual: 0.0,
            }],
            centralities: vec![CentralityState::Unavailable {
                labels: vec!["alpha".into()],
                reason: "none".into(),
            }],
            distance_matrices: vec![DistanceMatrix::new(vec!["alpha".into()], vec![0.0])],
            etv_dataset: EtvDataset {
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
        assert!(json.contains("\"etv_dataset\""));

        let _ = std::fs::remove_file(path);
    }
}
