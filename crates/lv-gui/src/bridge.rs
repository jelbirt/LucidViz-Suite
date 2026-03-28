//! Bridge between MatrixForge series output and AlignSpace pipeline input.
//!
//! This module lives in `lv-gui` (rather than in either pipeline crate) to
//! avoid a direct dependency from `mf-pipeline` → `as-pipeline`.

use anyhow::Result;
use as_pipeline::pipeline::mf_output_to_distance_matrix;
use as_pipeline::types::{
    AsDistancePipelineInput, CentralityMode, MdsConfig, MdsDimMode, NormalizationMode,
    ProcrustesMode,
};
use mf_pipeline::types::MfSeriesOutput;

/// Options for converting MF series output into an AS distance pipeline input.
#[derive(Debug, Clone)]
pub struct MfSeriesAsInputOptions {
    pub mds_config: MdsConfig,
    pub procrustes_mode: ProcrustesMode,
    pub mds_dims: MdsDimMode,
    pub normalize: bool,
    pub normalization_mode: NormalizationMode,
    pub target_range: f64,
    pub procrustes_scale: bool,
    pub centrality_mode: CentralityMode,
}

/// Convert MatrixForge series output into an AlignSpace multi-step distance input.
pub fn mf_series_output_to_as_input(
    output: &MfSeriesOutput,
    options: MfSeriesAsInputOptions,
) -> Result<AsDistancePipelineInput> {
    output.validate_for_as_input()?;
    let datasets = output
        .slices
        .iter()
        .map(|slice| {
            Ok((
                slice.label.clone(),
                mf_output_to_distance_matrix(
                    slice.output.labels.clone(),
                    &slice.output.similarity_matrix,
                    slice.output.n,
                    slice.output.sim_to_dist,
                )
                .map_err(|e| anyhow::anyhow!(e))?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(AsDistancePipelineInput {
        datasets,
        mds_config: options.mds_config,
        procrustes_mode: options.procrustes_mode,
        mds_dims: options.mds_dims,
        normalize: options.normalize,
        normalization_mode: options.normalization_mode,
        target_range: options.target_range,
        procrustes_scale: options.procrustes_scale,
        centrality_mode: options.centrality_mode,
    })
}
