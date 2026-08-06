# LVS Bug Audit Report

**Date:** 2026-04-04
**Auditor:** Claude Code
**Baseline:** 314 tests passing (76 + 8 + 3 + 4 + 5 + 22 + 11 + 50 + 2 + 7 + 6 + 25 + 39 + 4 + 41 + 11 = 314), clippy clean: yes, fmt clean: yes (nightly-only warnings on import granularity)

---

## Critical

No confirmed critical bugs. The UndoStack eviction path (Section 6) was analyzed and found to be **currently correct** — `cursor = self.stack.len()` executes after `remove(0)`, so cursor equals post-eviction length. However, this path has **zero test coverage**, making it a single line-reorder away from a panic. See [M10] below.

---

## High

**[H1] `crates/as-pipeline/src/procrustes.rs:229` — `procrustes_residual` uses transposed R for translation but untransposed R for point application.**
The translation is computed as `mu_b - ss * mu_a * R^T` but points are transformed as `x * R` (no transpose). In `procrustes()` (lines 109, 119) both use `R` consistently. The mismatch in `procrustes_residual` produces an incorrect residual score. This feeds into `choose_optimal_reference` (`pipeline.rs:377`), which selects the temporal reference slice based on summed residuals — a wrong residual ordering can select a suboptimal reference, causing drift accumulation across the entire temporal Procrustes chain.
**Fix:** Change line 229 from `r_mat.transpose()` to `&r_mat`. Add a cross-check test that `procrustes_residual` matches `procrustes().residual` for identical inputs.

**[H2] `crates/lv-wasm/src/lib.rs:91, 153` — Unchecked `result.coordinates[0]` panics in WASM if pipeline returns empty coordinates.**
Both `run_mf_to_coordinates` and `run_as` index `result.coordinates[0]` unconditionally. If the pipeline returns `Ok` with an empty `coordinates` vec (e.g., n=1 after stopword filtering), this panics. In WASM, a Rust panic calls `unreachable` — the module terminates silently with no JS-catchable error.
**Fix:** Replace with `result.coordinates.into_iter().next().ok_or_else(|| JsError::new("pipeline returned no coordinate slices"))?`.

**[H3] `crates/as-pipeline/src/mds/tsne.rs:222` — `p.ln()` on zero probability produces NaN that propagates to renderer.**
The entropy computation `-p * p.ln()` returns NaN when `p == 0.0` (since `0.0 * -Inf = NaN` per IEEE 754). This occurs when pairwise distances are very large relative to sigma. The NaN propagates through the perplexity binary search into the final coordinate array, producing an all-NaN LIS buffer.
**Fix:** Replace with `.map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })`.

**[H4] `crates/lv-app/src/session.rs:22-43` — `overrides`, `as_input_source`, and `play_state` are never saved in SessionSnapshot.**
`AppState.overrides: HashMap<String, ObjectOverride>` (per-node shape/color/size customizations) is absent from `SessionSnapshot`. Users lose all visual overrides on session save/load with no warning. Similarly, `as_input_source` (Dataset vs. MatrixForge mode) silently resets to `Dataset` on load, and `play_state` always restores to `Playing`.
**Fix:** Add these fields to `SessionSnapshot` with `#[serde(default)]`. `ObjectOverride` needs `Serialize`/`Deserialize` derives (currently only `Clone, Debug` at `state.rs:137`).

**[H5] `crates/lv-app/src/session.rs:22-131` — Most session struct fields lack `#[serde(default)]`, causing hard deserialization failure on version skew.**
Only 2 fields across all session structs have `#[serde(default)]` (`LisConfigSnapshot.easing` and `AudioSnapshot.mapping`). All others — including `AudioSnapshot.graduated`, `semitone_range`, `beats`, `hold_slices` and all of `EgoSnapshot`, `ExportSnapshot` — will cause `serde_json::from_slice` to return a hard error when loading a session saved before those fields existed. The session file becomes permanently unreadable with no migration path.
**Fix:** Add `#[serde(default)]` to every field in `SessionSnapshot`, `EgoSnapshot`, `ExportSnapshot`, `AudioSnapshot`, and `LisConfigSnapshot` that has a reasonable default value.

---

## Medium

**[M1] `crates/lv-audio/src/scheduler.rs:151` — `BetweennessPitchClosenessVelocity` uses raw `cluster_value` as velocity, not closeness centrality.**
The variant name and doc comment claim `cluster_value` carries closeness centrality, but per `lv-data/src/schema.rs:139`, `cluster_value` is "cluster membership value; >= 0" — an arbitrary scalar from column 17 of the input spreadsheet. Datasets with cluster indices like 2, 5, or 10 all clamp to `vel = 127` via `.clamp(0.0, 1.0) * 127.0`, producing maximum velocity regardless of intent.
**Fix:** Either populate a dedicated `closeness_centrality` field from AS pipeline output, or rename the variant and normalize `cluster_value` against observed dataset range.

**[M2] `crates/as-pipeline/src/structural_eq.rs:158` — `(var_a * var_b).sqrt()` can receive a negative argument, producing NaN.**
For near-constant profiles, the two-pass variance formula `sum_sq/n - mean^2` can yield a small negative value due to catastrophic cancellation. The subsequent `.sqrt()` returns NaN, which bypasses the `denom < 1e-15` guard (NaN is not `< 1e-15`).
**Fix:** `let denom = (var_a.max(0.0) * var_b.max(0.0)).sqrt();`

**[M3] `crates/as-pipeline/src/mds/classical.rs:71` — NaN eigenvalues not filtered before coordinate use.**
`f64::NAN.max(0.0)` returns NaN in Rust (max propagates NaN), so the eigenvalue clamp does not sanitize NaN. The `scale = lam.sqrt()` at line 72 then produces NaN, contaminating the coordinate array.
**Fix:** `let lam = if lam.is_finite() { lam.max(0.0) } else { 0.0 };`

**[M4] `crates/lv-app/src/main.rs:1387-1397` — `EdgeUniforms.viewport_size` is stale for one frame after window resize.**
The resize handler reconfigures the surface, depth texture, scene texture, and camera aspect ratio, but does NOT write the edge uniform buffer. `viewport_size` is only updated on the next `render()` call. One frame renders with old viewport dimensions against new surface dimensions, causing edge-width artifacts on resize drag.
**Fix:** Write the edge uniform buffer synchronously inside `resize()` after updating `self.window_size`.

**[M5] `crates/lv-gui/src/panels/file_loader.rs:85-93` — Stale recent-file paths passed directly to XLSX/JSON reader without existence check.**
The Reload button correctly gates on `path.exists()` (line 53), but recent-file click handlers pass the path to parsers with no existence guard. A deleted file produces a confusing parser-level I/O error instead of a clear "file not found" message.
**Fix:** Add `if !path.exists()` guard before the load call, matching the Reload button pattern.

**[M6] `crates/lv-app/src/main.rs:1657-1658` — No `Focused(false)` handler to clear modifier state causes "stuck modifier" bugs.**
The app stores `modifiers: ModifiersState` updated on `ModifiersChanged`, but never clears it on focus loss. If the user presses Ctrl, alt-tabs, releases Ctrl in another app, then returns, `ctrl` remains true. Pressing `Z` fires undo instead of typing, `S` fires save instead of zooming.
**Fix:** Add `WindowEvent::Focused(false) => { self.modifiers = ModifiersState::empty(); }`.

**[M7] `crates/lv-app/src/main.rs:1042` — Export `cancel_flag` may never be read by `lv-export`.**
The `AtomicBool` cancel flag is created and passed to the export thread, but grep of `lv-export/src/` found no matches for `cancel_flag`, `AtomicBool`, `load`, or `Ordering`. If the flag is never checked, the Cancel button is a no-op and long exports cannot be interrupted.
**Fix:** Verify and add `cancel_flag.load(Ordering::Relaxed)` checks in the export frame loop.

**[M8] `crates/lv-app/src/notifications.rs:43, 81` — `Notification::warn` and `notify_error` are dead code (`#[allow(dead_code)]`).**
The warning notification tier exists but is never called. This confirms that actionable warnings (prefs save failures, stale-path errors) are not routed to the notification system.
**Fix:** Remove `#[allow(dead_code)]` and wire these functions to the error sites identified in this report.

---

## Low / Documentation

**[L1] CLAUDE.md — MDS threshold documented as "N < 500" but actual dispatch is `n < 800` at `crates/as-pipeline/src/mds/mod.rs:34`.**
Documentation drift only. No functional impact, but misleads future auditors and contributors.

**[L2] `crates/lv-audio/src/scheduler.rs:155` — `abs()` in `ClusterToChannel` masks negative input that validation should reject.**
Validation at `lv-data/src/validation.rs:92` rejects `cluster_value < 0`, so `abs()` is a defensive no-op. But it silently remaps invalid input rather than surfacing a contract violation. Replace with `debug_assert!(row.cluster_value >= 0.0)`.

**[L3] `crates/as-pipeline/src/procrustes.rs:101-105` — Procrustes `ss = 1.0` fallback for degenerate source cloud is undocumented.**
When `frob_a_sq <= 1e-15` (all source points coincident after centering), scale falls back to `ss = 1.0`. Mathematically arbitrary but avoids division by zero. Callers have no way to detect the degenerate case.

**[L4] `crates/lv-gui/src/state.rs:313-321` — UndoStack eviction path has zero test coverage.**
The current implementation is correct (cursor is set after `remove(0)`), but `Vec::remove(0)` is O(N) and any line reorder would introduce an out-of-bounds panic. No test exercises `max_depth` eviction. Consider switching to `VecDeque` and adding an eviction test.

**[L5] `crates/as-pipeline/src/mds/mod.rs:67` — `resolve_dims` underflows on `n=1` but is guarded by `run_mds` check at line 27-29.**
The function is private and only called from `run_mds` after the `n < 2` guard. Safe today, but adding an inline comment noting the `n >= 2` invariant would prevent regressions if the function is ever made public.

**[L6] `crates/lv-app/src/main.rs:1321-1325` — `prefs.save()` failure after dataset load is logged but not surfaced as a notification.**
The user sees the recent file appear in the menu, but it will be absent after restart with no explanation.

**[L7] `crates/lv-app/src/main.rs:1506-1516` — `get_current_texture()` failure silently dropped via `.ok()`.**
Surface acquisition errors (minimized window, driver error) are not logged, resulting in a frozen viewport with no diagnostic.

---

## Confirmed Correct (Sections Where No Bug Found)

- **Section 3 — `resolve_dims` edge case:** Guarded by `run_mds` `n < 2` check. Private function, single call site. WASM wrappers route through `run_distance_pipeline` → `run_mds`. No exposure.
- **Section 6 — UndoStack push/undo/redo logic:** The eviction path is currently correct (`cursor = stack.len()` after `remove(0)`). Flagged as [L4] for missing test coverage only.
- **Section 10 — Feature gate correctness:** All audio/export/video-export gates are properly applied. `cargo build -p lv-app --no-default-features` succeeds. Two pre-existing dead-code warnings in `notifications.rs` are unrelated to feature gating.
- **Section 12 — mpsc channel lifecycle:** `poll_jobs` correctly guards against re-polling after `Done`. `ExportJob` uses two distinct channels. Job drop causes background thread to see `SendError` and terminate naturally.
- **Section 5 — BeatMapping/SonificationMapping enum ordering:** Variants are identical in both enums. Conversion at `main.rs:806-816` uses structural `match`, not integer cast. No silent mismatch.

---

## Summary by Severity

| Severity | Count | Sections |
|----------|-------|----------|
| Critical | 0 | — |
| High | 5 | 1, 4, 8, 7, 11 |
| Medium | 8 | 5, 9, 11, 12, 13, 14, 15 |
| Low | 7 | 2, 3, 5, 6, 11, 14, 15 |

### Recommended Fix Priority

1. **[H5]** Add `#[serde(default)]` to all session struct fields — prevents data loss on version upgrade
2. **[H3]** Fix t-SNE `p.ln()` on zero — prevents NaN coordinate propagation
3. **[H1]** Fix `procrustes_residual` R^T mismatch — corrects temporal reference selection
4. **[H2]** Guard `result.coordinates[0]` in WASM — prevents silent module termination
5. **[H4]** Add `overrides`/`as_input_source`/`play_state` to SessionSnapshot — prevents silent data loss
6. **[M1-M8]** Address medium findings in order listed
