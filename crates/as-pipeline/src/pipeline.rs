//! Main AlignSpace pipeline orchestration.

use anyhow::{bail, Result};
use ndarray::Array2;

use lv_data::schema::EtvDataset;

use crate::centrality::compute_centrality;
use crate::mds::run_mds;
use crate::normalize::normalize_coordinates;
use crate::procrustes::procrustes;
use crate::structural_eq::compute_se_matrix;
use crate::types::{
    AsPipelineInput, AsPipelineResult, CentralityReport, MdsCoordinates, ProcrustesMode,
    ProcrustesResult, SeMatrix,
};

/// Run the full AlignSpace pipeline.
///
/// # Sequence
/// 1. For each (name, adjacency) in `input.datasets`:
///    a. Compute SE matrix
///    b. Run MDS → `MdsCoordinates`
///    c. Compute centrality
/// 2. Procrustes alignment (if requested)
/// 3. Normalize coordinates (if requested)
/// 4. Build `EtvDataset` from final coordinates
pub fn run_pipeline(input: &AsPipelineInput) -> Result<AsPipelineResult> {
    if input.datasets.is_empty() {
        bail!("AsPipelineInput has no datasets");
    }

    let mut se_matrices: Vec<SeMatrix> = Vec::new();
    let mut raw_coords: Vec<MdsCoordinates> = Vec::new();
    let mut centralities: Vec<CentralityReport> = Vec::new();

    for (name, adj) in &input.datasets {
        // Compute SE matrix.
        let se = compute_se_matrix(adj, input.labels.clone());
        se_matrices.push(se.clone());

        // Run MDS.
        let coords = run_mds(&se, &input.mds_config, input.mds_dims)?;
        raw_coords.push(coords);

        // Compute centrality.
        let centrality = compute_centrality(adj, &input.labels);
        centralities.push(centrality);

        let _ = name; // used for future output naming
    }

    // Procrustes alignment.
    let procrustes_results: Vec<ProcrustesResult> = match input.procrustes_mode {
        ProcrustesMode::None => {
            // No alignment — create identity procrustes results.
            raw_coords
                .iter()
                .map(|c| ProcrustesResult {
                    aligned: c.clone(),
                    rotation: identity_rotation(c.dims),
                    scale: 1.0,
                    translation: vec![0.0; c.dims],
                    residual: 0.0,
                })
                .collect()
        }
        ProcrustesMode::TimeSeries => {
            let mut aligned_coords: Vec<MdsCoordinates> = raw_coords.clone();
            let mut proc_results: Vec<ProcrustesResult> = Vec::new();

            // First step: identity.
            proc_results.push(ProcrustesResult {
                aligned: aligned_coords[0].clone(),
                rotation: identity_rotation(aligned_coords[0].dims),
                scale: 1.0,
                translation: vec![0.0; aligned_coords[0].dims],
                residual: 0.0,
            });

            // Align each subsequent step to the previous.
            for (i, source) in raw_coords.iter().enumerate().skip(1) {
                let target = aligned_coords[i - 1].clone();
                let res = procrustes(source, &target, input.procrustes_scale)?;
                aligned_coords[i] = res.aligned.clone();
                proc_results.push(res);
            }
            proc_results
        }
        ProcrustesMode::OptimalChoice => {
            // Find the pair with the best (lowest) alignment residual and use that
            // as the reference; for now fall back to step 0 as reference.
            // Full optimal implementation would try all pairs; this is a simplification.
            let reference = raw_coords[0].clone();
            let mut proc_results: Vec<ProcrustesResult> = Vec::new();
            proc_results.push(ProcrustesResult {
                aligned: reference.clone(),
                rotation: identity_rotation(reference.dims),
                scale: 1.0,
                translation: vec![0.0; reference.dims],
                residual: 0.0,
            });
            for (idx, source) in raw_coords.iter().enumerate().skip(1) {
                let res = procrustes(source, &reference, input.procrustes_scale)?;
                let _ = idx;
                proc_results.push(res);
            }
            proc_results
        }
    };

    // Extract final aligned coordinates.
    let mut coordinates: Vec<MdsCoordinates> = procrustes_results
        .iter()
        .map(|r| r.aligned.clone())
        .collect();

    // Normalize if requested.
    if input.normalize {
        for coords in coordinates.iter_mut() {
            normalize_coordinates(coords, input.target_range);
        }
    }

    // Build a minimal EtvDataset from the last time step's coordinates.
    let etv_dataset = build_etv_dataset(&coordinates, &input.datasets);

    Ok(AsPipelineResult {
        coordinates,
        procrustes: procrustes_results,
        centralities,
        se_matrices,
        etv_dataset,
    })
}

// ---------------------------------------------------------------------------
// MF → AS bridge
// ---------------------------------------------------------------------------

/// Similarity-to-distance conversion methods (mirrors MfConfig).
#[derive(Debug, Clone, Copy)]
pub enum SimToDistMethod {
    /// d = 1 - s
    Linear,
    /// d = sqrt(1 - s²)
    Cosine,
    /// d = -ln(s)  (s must be > 0)
    Info,
}

/// Convert an MF output similarity matrix into an AS-compatible SeMatrix.
///
/// `sim[i,j]` values are assumed in [0, 1].
pub fn mf_output_to_se_matrix(
    labels: Vec<String>,
    similarity_matrix: &[f64],
    n: usize,
    method: SimToDistMethod,
) -> SeMatrix {
    assert_eq!(similarity_matrix.len(), n * n);
    let data: Vec<f64> = similarity_matrix
        .iter()
        .map(|&s| sim_to_dist(s, method))
        .collect();
    SeMatrix::new(labels, data)
}

fn sim_to_dist(s: f64, method: SimToDistMethod) -> f64 {
    match method {
        SimToDistMethod::Linear => (1.0 - s).max(0.0),
        SimToDistMethod::Cosine => (1.0 - s * s).max(0.0).sqrt(),
        SimToDistMethod::Info => {
            if s <= 0.0 {
                f64::INFINITY
            } else {
                -s.ln()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn identity_rotation(dims: usize) -> Vec<f64> {
    let mut r = vec![0.0f64; dims * dims];
    for i in 0..dims {
        r[i * dims + i] = 1.0;
    }
    r
}

fn build_etv_dataset(
    coordinates: &[MdsCoordinates],
    datasets: &[(String, Array2<f64>)],
) -> EtvDataset {
    use lv_data::schema::{EtvRow, EtvSheet, ShapeKind};

    if coordinates.is_empty() {
        return EtvDataset {
            source_path: None,
            sheets: vec![],
            all_labels: vec![],
        };
    }

    // Create one sheet per time step.
    let sheets: Vec<EtvSheet> = coordinates
        .iter()
        .enumerate()
        .map(|(step_idx, coords)| {
            let name = datasets
                .get(step_idx)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("Step_{}", step_idx + 1));

            let rows: Vec<EtvRow> = coords
                .labels
                .iter()
                .enumerate()
                .map(|(i, label)| {
                    let x = if coords.dims >= 1 {
                        coords.data[i * coords.dims]
                    } else {
                        0.0
                    };
                    let y = if coords.dims >= 2 {
                        coords.data[i * coords.dims + 1]
                    } else {
                        0.0
                    };
                    EtvRow {
                        label: label.clone(),
                        shape: ShapeKind::Sphere,
                        x,
                        y,
                        ..Default::default()
                    }
                })
                .collect();

            EtvSheet {
                name,
                sheet_index: step_idx,
                rows,
                edges: vec![],
            }
        })
        .collect();

    let all_labels = if let Some(first) = coordinates.first() {
        first.labels.clone()
    } else {
        vec![]
    };

    EtvDataset {
        source_path: None,
        sheets,
        all_labels,
    }
}
