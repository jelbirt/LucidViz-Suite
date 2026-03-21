# Audit Summary

This file tracks the current consolidated audit status for the AS -> LV path.

## Implemented Since 2026-03-21

- Critical/high follow-up fixes landed:
  - LIS frames now carry stable labels alongside GPU instances.
  - LV ego filtering, picking, and ego-edge overlays now use the active frame instead of `sheets.first()`.
  - GUI AlignSpace output now preserves the core AS artifact contract in `as_result.json` and writes a separate `etv_dataset.json` for LV reload.
- Medium/low follow-up fixes landed:
  - `EtvDataset::all_labels` is now canonicalized from sheet rows during normal JSON/XLSX loading paths, and dataset validation rejects stale/non-canonical `all_labels`.
  - GUI dataset-to-AS ingestion now rebuilds labels from sheet rows instead of preserving stale extras from `all_labels`.
  - AlignSpace now supports `NormalizationMode::{Independent, Global}` so temporal runs can preserve relative scale across slices when needed.
  - AlignSpace now supports `ProcrustesMode::TimeSeriesAnchored` so time-series alignment can use slice 0 as a fixed reference instead of chaining slice-to-slice.
  - LIS buffer size estimation now uses `all_labels.len()` and accounts for single-sheet static animation buffers.
  - `LisFrame.slice_index` docs and frame-count comments/tests were updated to match implementation.
- Directed-edge follow-up fixes landed:
  - GUI dataset -> AS ingestion now preserves directed adjacency instead of symmetrizing edges.
  - AS export now emits one `EdgeRow` per nonzero directed adjacency entry, so reciprocal ties remain distinct in LV datasets.
  - LV ego visibility and ego-edge overlays now support `Incoming`, `Outgoing`, and `Both` direction modes, with `Both` as the backward-compatible default.

## Remaining Notable Limitation

- AlignSpace centrality is still intentionally undirected. Structural equivalence and LV/runtime edge handling now preserve directedness, but `compute_centrality` still consumes only the upper triangle of the adjacency matrix for backward compatibility.

## Primary Evidence Files

- `crates/as-pipeline/src/types.rs`
- `crates/as-pipeline/src/pipeline.rs`
- `crates/as-pipeline/src/normalize.rs`
- `crates/lv-data/src/schema.rs`
- `crates/lv-data/src/validation.rs`
- `crates/lv-data/src/json_io.rs`
- `crates/lv-data/src/xlsx_reader.rs`
- `crates/lv-renderer/src/lis.rs`
- `crates/lv-app/src/main.rs`
- `crates/lv-gui/src/as_panel.rs`
- `tests/integration/as_pipeline_e2e.rs`
- `tests/integration/mf_pipeline_e2e.rs`
- `tests/integration/full_render_cycle.rs`
- `tests/integration/round_trip_xlsx.rs`

## Related Detailed Notes

- `audit03212026.md`
