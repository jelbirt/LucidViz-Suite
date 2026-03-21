//! Main AlignSpace pipeline orchestration.

use anyhow::{bail, Result};
use ndarray::Array2;

use lv_data::schema::EtvDataset;
use lv_data::SimToDistMethod;

use crate::centrality::compute_centrality;
use crate::mds::run_mds;
use crate::normalize::{normalize_coordinate_series, normalize_coordinates};
use crate::procrustes::procrustes;
use crate::structural_eq::compute_se_matrix;
use crate::types::{
    AsDistancePipelineInput, AsPipelineInput, AsPipelineResult, CentralityState, DistanceMatrix,
    MdsCoordinates, NormalizationMode, ProcrustesMode, ProcrustesResult,
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

    let mut distance_matrices: Vec<DistanceMatrix> = Vec::new();
    let mut raw_coords: Vec<MdsCoordinates> = Vec::new();
    let mut centralities: Vec<CentralityState> = Vec::new();

    for (name, adj) in &input.datasets {
        // Compute SE matrix.
        let se = compute_se_matrix(adj, input.labels.clone());
        distance_matrices.push(se.clone());

        // Run MDS.
        let coords = run_mds(&se, &input.mds_config, input.mds_dims)?;
        raw_coords.push(coords);

        // Compute centrality.
        let centrality = compute_centrality(adj, &input.labels);
        centralities.push(CentralityState::Computed(centrality));

        let _ = name; // used for future output naming
    }

    let procrustes_results =
        align_coordinates(&raw_coords, input.procrustes_mode, input.procrustes_scale)?;

    // Extract final aligned coordinates.
    let mut coordinates: Vec<MdsCoordinates> = procrustes_results
        .iter()
        .map(|r| r.aligned.clone())
        .collect();

    // Normalize if requested.
    if input.normalize {
        normalize_output_coordinates(
            &mut coordinates,
            input.normalization_mode,
            input.target_range,
        );
    }

    // Build a minimal EtvDataset from the last time step's coordinates.
    let etv_dataset = build_etv_dataset(&coordinates, &input.datasets);

    Ok(AsPipelineResult {
        coordinates,
        procrustes: procrustes_results,
        centralities,
        distance_matrices,
        etv_dataset,
    })
}

/// Run AlignSpace from precomputed distance matrices instead of adjacency inputs.
pub fn run_distance_pipeline(input: &AsDistancePipelineInput) -> Result<AsPipelineResult> {
    if input.datasets.is_empty() {
        bail!("AsDistancePipelineInput has no datasets");
    }

    let distance_matrices: Vec<DistanceMatrix> =
        input.datasets.iter().map(|(_, se)| se.clone()).collect();
    let raw_coords: Vec<MdsCoordinates> = distance_matrices
        .iter()
        .map(|se| run_mds(se, &input.mds_config, input.mds_dims))
        .collect::<Result<_>>()?;

    let labels = distance_matrices
        .first()
        .map(|se| se.labels.clone())
        .unwrap_or_default();
    let centralities: Vec<CentralityState> = distance_matrices
        .iter()
        .map(|_| unavailable_centrality_state(&labels))
        .collect();

    let procrustes_results =
        align_coordinates(&raw_coords, input.procrustes_mode, input.procrustes_scale)?;

    let mut coordinates: Vec<MdsCoordinates> = procrustes_results
        .iter()
        .map(|r| r.aligned.clone())
        .collect();

    if input.normalize {
        normalize_output_coordinates(
            &mut coordinates,
            input.normalization_mode,
            input.target_range,
        );
    }

    let dataset_names: Vec<(String, Array2<f64>)> = input
        .datasets
        .iter()
        .map(|(name, se)| (name.clone(), Array2::<f64>::zeros((se.n, se.n))))
        .collect();
    let etv_dataset = build_etv_dataset(&coordinates, &dataset_names);

    Ok(AsPipelineResult {
        coordinates,
        procrustes: procrustes_results,
        centralities,
        distance_matrices,
        etv_dataset,
    })
}

// ---------------------------------------------------------------------------
// MF → AS bridge
// ---------------------------------------------------------------------------

/// Convert an MF output similarity matrix into an AS-compatible distance matrix.
///
/// `sim[i,j]` values are assumed in [0, 1].
pub fn mf_output_to_distance_matrix(
    labels: Vec<String>,
    similarity_matrix: &[f64],
    n: usize,
    method: SimToDistMethod,
) -> DistanceMatrix {
    assert_eq!(similarity_matrix.len(), n * n);
    let data: Vec<f64> = similarity_matrix
        .iter()
        .map(|&s| sim_to_dist(s, method))
        .collect();
    DistanceMatrix::new(labels, data)
}

/// Legacy bridge name retained for compatibility.
pub fn mf_output_to_se_matrix(
    labels: Vec<String>,
    similarity_matrix: &[f64],
    n: usize,
    method: SimToDistMethod,
) -> DistanceMatrix {
    mf_output_to_distance_matrix(labels, similarity_matrix, n, method)
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

fn unavailable_centrality_state(labels: &[String]) -> CentralityState {
    CentralityState::Unavailable {
        labels: labels.to_vec(),
        reason:
            "Centrality is unavailable for precomputed distance inputs without an adjacency graph"
                .to_string(),
    }
}

fn align_coordinates(
    raw_coords: &[MdsCoordinates],
    procrustes_mode: ProcrustesMode,
    procrustes_scale: bool,
) -> Result<Vec<ProcrustesResult>> {
    let results = match procrustes_mode {
        ProcrustesMode::None => raw_coords
            .iter()
            .map(|c| ProcrustesResult {
                aligned: c.clone(),
                rotation: identity_rotation(c.dims),
                scale: 1.0,
                translation: vec![0.0; c.dims],
                residual: 0.0,
            })
            .collect(),
        ProcrustesMode::TimeSeries => {
            let mut aligned_coords: Vec<MdsCoordinates> = raw_coords.to_vec();
            let mut proc_results: Vec<ProcrustesResult> = Vec::new();

            proc_results.push(ProcrustesResult {
                aligned: aligned_coords[0].clone(),
                rotation: identity_rotation(aligned_coords[0].dims),
                scale: 1.0,
                translation: vec![0.0; aligned_coords[0].dims],
                residual: 0.0,
            });

            for (i, source) in raw_coords.iter().enumerate().skip(1) {
                let target = aligned_coords[i - 1].clone();
                let res = procrustes(source, &target, procrustes_scale)?;
                aligned_coords[i] = res.aligned.clone();
                proc_results.push(res);
            }
            proc_results
        }
        ProcrustesMode::TimeSeriesAnchored => {
            let reference = raw_coords[0].clone();
            let mut proc_results: Vec<ProcrustesResult> = Vec::new();

            proc_results.push(ProcrustesResult {
                aligned: reference.clone(),
                rotation: identity_rotation(reference.dims),
                scale: 1.0,
                translation: vec![0.0; reference.dims],
                residual: 0.0,
            });

            for source in raw_coords.iter().skip(1) {
                proc_results.push(procrustes(source, &reference, procrustes_scale)?);
            }

            proc_results
        }
        ProcrustesMode::OptimalChoice => {
            let reference_idx = choose_optimal_reference(raw_coords, procrustes_scale)?;
            let reference = raw_coords[reference_idx].clone();
            let mut proc_results: Vec<ProcrustesResult> = Vec::new();
            for (idx, source) in raw_coords.iter().enumerate() {
                if idx == reference_idx {
                    proc_results.push(ProcrustesResult {
                        aligned: reference.clone(),
                        rotation: identity_rotation(reference.dims),
                        scale: 1.0,
                        translation: vec![0.0; reference.dims],
                        residual: 0.0,
                    });
                } else {
                    proc_results.push(procrustes(source, &reference, procrustes_scale)?);
                }
            }
            proc_results
        }
    };

    Ok(results)
}

fn normalize_output_coordinates(
    coordinates: &mut [MdsCoordinates],
    normalization_mode: NormalizationMode,
    target_range: f64,
) {
    match normalization_mode {
        NormalizationMode::Independent => {
            for coords in coordinates.iter_mut() {
                normalize_coordinates(coords, target_range);
            }
        }
        NormalizationMode::Global => normalize_coordinate_series(coordinates, target_range),
    }
}

fn choose_optimal_reference(
    raw_coords: &[MdsCoordinates],
    procrustes_scale: bool,
) -> Result<usize> {
    let mut best: Option<(usize, f64)> = None;

    for (candidate_idx, candidate) in raw_coords.iter().enumerate() {
        let mut total_residual = 0.0;
        let mut valid = true;

        for (other_idx, other) in raw_coords.iter().enumerate() {
            if candidate_idx == other_idx {
                continue;
            }

            match procrustes(other, candidate, procrustes_scale) {
                Ok(result) => {
                    total_residual += result.residual;
                }
                Err(_) => {
                    valid = false;
                    break;
                }
            }
        }

        if valid {
            match best {
                Some((_, best_residual)) if total_residual >= best_residual => {}
                _ => best = Some((candidate_idx, total_residual)),
            }
        }
    }

    if let Some((idx, _)) = best {
        Ok(idx)
    } else {
        bail!("OptimalChoice could not find a reference sharing enough labels with all time steps")
    }
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

    // Create one sheet per time step. For adjacency-backed inputs, preserve
    // directed edges exactly as they appear in the adjacency matrix.
    // Distance-backed inputs pass zero-adjacency placeholders, so they export
    // coordinate-only sheets with no edges.
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
                    let z = if coords.dims >= 3 {
                        coords.data[i * coords.dims + 2]
                    } else {
                        0.0
                    };
                    EtvRow {
                        label: label.clone(),
                        shape: ShapeKind::Sphere,
                        x,
                        y,
                        z,
                        ..Default::default()
                    }
                })
                .collect();

            let edges = datasets
                .get(step_idx)
                .map(|(_, adj)| adjacency_to_edges(&coords.labels, adj))
                .unwrap_or_default();

            EtvSheet {
                name,
                sheet_index: step_idx,
                rows,
                edges,
            }
        })
        .collect();

    EtvDataset {
        source_path: None,
        all_labels: EtvDataset::canonical_all_labels_from_sheets(&sheets),
        sheets,
    }
}

fn adjacency_to_edges(labels: &[String], adj: &Array2<f64>) -> Vec<lv_data::schema::EdgeRow> {
    let n = labels.len().min(adj.nrows()).min(adj.ncols());
    let mut edges = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let strength = adj[[i, j]];
            if strength != 0.0 {
                edges.push(lv_data::schema::EdgeRow {
                    from: labels[i].clone(),
                    to: labels[j].clone(),
                    strength,
                });
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::{align_coordinates, build_etv_dataset, unavailable_centrality_state};
    use crate::types::{CentralityState, MdsAlgorithm, MdsCoordinates, ProcrustesMode};
    use ndarray::array;

    #[test]
    fn build_etv_dataset_preserves_z_and_adjacency_edges() {
        let coords = MdsCoordinates::new(
            vec!["alpha".into(), "beta".into()],
            vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            3,
            0.0,
            MdsAlgorithm::Classical,
        );
        let datasets = vec![("T1".to_string(), array![[0.0, 0.75], [0.75, 0.0]])];

        let etv = build_etv_dataset(&[coords], &datasets);

        assert_eq!(etv.sheets.len(), 1);
        assert_eq!(etv.sheets[0].rows[0].z, 3.0);
        assert_eq!(etv.sheets[0].rows[1].z, -3.0);
        assert_eq!(etv.sheets[0].edges.len(), 2);
        assert_eq!(etv.sheets[0].edges[0].from, "alpha");
        assert_eq!(etv.sheets[0].edges[0].to, "beta");
        assert!((etv.sheets[0].edges[0].strength - 0.75).abs() < 1e-12);
        assert_eq!(etv.sheets[0].edges[1].from, "beta");
        assert_eq!(etv.sheets[0].edges[1].to, "alpha");
        assert!((etv.sheets[0].edges[1].strength - 0.75).abs() < 1e-12);
    }

    #[test]
    fn build_etv_dataset_preserves_directed_edges() {
        let coords = MdsCoordinates::new(
            vec!["alpha".into(), "beta".into()],
            vec![1.0, 2.0, -1.0, -2.0],
            2,
            0.0,
            MdsAlgorithm::Classical,
        );
        let datasets = vec![("T1".to_string(), array![[0.0, 0.75], [0.0, 0.0]])];

        let etv = build_etv_dataset(&[coords], &datasets);

        assert_eq!(etv.sheets[0].edges.len(), 1);
        assert_eq!(etv.sheets[0].edges[0].from, "alpha");
        assert_eq!(etv.sheets[0].edges[0].to, "beta");
        assert!((etv.sheets[0].edges[0].strength - 0.75).abs() < 1e-12);
    }

    #[test]
    fn build_etv_dataset_exports_reciprocal_edges_separately() {
        let coords = MdsCoordinates::new(
            vec!["alpha".into(), "beta".into()],
            vec![1.0, 2.0, -1.0, -2.0],
            2,
            0.0,
            MdsAlgorithm::Classical,
        );
        let datasets = vec![("T1".to_string(), array![[0.0, 0.75], [0.5, 0.0]])];

        let etv = build_etv_dataset(&[coords], &datasets);

        assert_eq!(etv.sheets[0].edges.len(), 2);
        assert_eq!(etv.sheets[0].edges[0].from, "alpha");
        assert_eq!(etv.sheets[0].edges[0].to, "beta");
        assert_eq!(etv.sheets[0].edges[1].from, "beta");
        assert_eq!(etv.sheets[0].edges[1].to, "alpha");
    }

    #[test]
    fn optimal_choice_uses_lowest_total_residual_reference() {
        let coords = vec![
            MdsCoordinates::new(
                vec!["a".into(), "b".into(), "c".into()],
                vec![0.0, 0.0, 1.0, 0.0, 0.1, 0.9],
                2,
                0.0,
                MdsAlgorithm::Classical,
            ),
            MdsCoordinates::new(
                vec!["a".into(), "b".into(), "c".into()],
                vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.8],
                2,
                0.0,
                MdsAlgorithm::Classical,
            ),
            MdsCoordinates::new(
                vec!["a".into(), "b".into(), "c".into()],
                vec![0.0, 0.0, 1.0, 0.0, 0.9, 0.6],
                2,
                0.0,
                MdsAlgorithm::Classical,
            ),
        ];

        let results = align_coordinates(&coords, ProcrustesMode::OptimalChoice, false)
            .expect("optimal choice should align");

        assert_eq!(results[1].residual, 0.0);
        assert!(results[0].residual > 0.0);
        assert!(results[2].residual > 0.0);
    }

    #[test]
    fn unavailable_centrality_state_is_explicit() {
        let state = unavailable_centrality_state(&["a".to_string(), "b".to_string()]);

        match state {
            CentralityState::Unavailable { labels, reason } => {
                assert_eq!(labels, vec!["a", "b"]);
                assert!(reason.contains("precomputed distance"));
            }
            CentralityState::Computed(_) => panic!("expected unavailable state"),
        }
    }
}
