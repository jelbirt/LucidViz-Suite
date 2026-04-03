## LVS Research & Development Roadmap

### Tier 1: Robustness & Safety (low effort, high impact)

**1. Harden NaN-panicking unwraps in hot paths**
- `as-pipeline/src/mds/multilevel.rs:220` — `partial_cmp().unwrap()` on user-supplied distances will panic on NaN input
- `as-pipeline/src/centrality.rs:515-517` — same NaN hazard plus empty-iterator panic on zero-node graphs
- These are the most likely runtime panics in production. Replacing with `unwrap_or(Ordering::Equal)` or early NaN-rejection at the pipeline entry point would close the gap.

**2. Test coverage for lv-renderer internals**
- `camera.rs`, `instanced.rs`, `edge_renderer.rs`, `pipelines.rs`, and all `shapes/` mesh builders have zero unit tests. The headless golden-hash test catches regressions at the integration level, but bugs in individual mesh geometry or camera projection math won't be isolated.

**3. Mutex-poison resilience in mf-pipeline**
- `stopwords.rs:21` — `STOPWORD_CACHE.lock().unwrap()` cascades panics across threads. Switching to `.lock().unwrap_or_else(|e| e.into_inner())` or a poison-free mutex (e.g., `parking_lot::Mutex`) prevents cascading failures.

---

### Tier 2: Feature Completion (medium effort)

**4. Wire up the `native-io` feature gates in as-pipeline and mf-pipeline**
- Both `Cargo.toml` files declare `native-io` with a comment "gate file I/O" but no `#[cfg(feature = "native-io")]` guards exist in source. This is a prerequisite for clean WASM builds — file I/O paths need to be gated so the pipeline crates compile for wasm32 without pulling in filesystem APIs.

**5. WASM beyond scaffold**
- Currently the WASM story is: crates compile for `wasm32-unknown-unknown`, but there are zero `#[wasm_bindgen]` exports, no `web-sys` canvas bindings, no WebGPU renderer path. The workspace already declares `wasm-bindgen`, `web-sys`, and `js-sys` as pinned deps.
- **Recommended approach:** Start with a `lv-wasm` crate that exposes the MF->AS pipeline as wasm-bindgen functions (text in -> 3D coordinates out). This is achievable without tackling the renderer. A web-based 3D viewer (three.js / WebGPU) can consume the coordinates separately.
- The renderer path is harder — `wgpu` supports WebGPU but `winit` + `egui-winit` need web event loop integration.

**6. Session management UX**
- `session_panel.rs` lacks delete/rename actions, no loading feedback (spinner), and saved sessions have no metadata preview. This is user-facing friction.

---

### Tier 3: Research Extensions (higher effort, high value)

**7. Streaming / incremental pipeline**
- The current MF->AS pipeline is batch: load corpus -> full co-occurrence -> full MDS. For large corpora, an incremental mode where new documents update the existing embedding (via online MDS or landmark-based projection) would enable real-time exploration.

**8. Additional MDS methods / dimensionality reduction**
- The codebase has Classical, Pivot, SMACOF, and Multilevel MDS. Adding t-SNE or UMAP as alternative embedders (with the same normalize -> Procrustes -> render pipeline) would let researchers compare layout algorithms on the same data.

**9. Directed-graph layout improvements**
- The pipeline preserves directed `EdgeRow { from, to }` semantics end-to-end, and the renderer/GUI support direction-based filtering. But the MDS embedding itself is undirected (symmetric distance matrix). Exploring asymmetric MDS or force-directed layouts that encode directionality in spatial position would be a meaningful research contribution.

**10. Temporal animation interpolation modes**
- LIS currently does linear interpolation between time slices. Adding easing functions (ease-in/out, cubic, spring) and configurable transition durations would improve temporal visualization quality. The `LisConfig` struct already exists and round-trips through sessions.

**11. Audio sonification depth**
- `lv-audio` has a graduated mapping system and MIDI engine, but there's no UI for loading soundbanks and no web audio fallback for WASM. Expanding the audio dimension (e.g., mapping centrality to pitch, cluster membership to timbre) would strengthen the multimodal angle.

---

### Tier 4: Infrastructure

**12. ffmpeg export integration test**
- The video export pipeline is fully implemented but has zero test coverage. A CI-friendly test that mocks or uses a minimal ffmpeg invocation would catch regressions in the pipe protocol.

**13. GUI panel tests**
- Most `lv-gui` panels (`cluster_filter`, `lis_controls`, `shape_overrides`, `file_loader`, `export_panel`, `audio_panel`, `session_panel`) have no unit tests. State-mutation logic in these panels could be extracted and tested without requiring an egui context.

**14. Documentation for scientific parameters**
- No tooltips or context-help exist in the UI for parameters like SMACOF tolerance, PMI window size, Procrustes alignment mode, CRF values, etc. Adding these would lower the barrier for non-expert users.
