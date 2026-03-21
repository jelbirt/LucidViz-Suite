# Lucid Visualization Suite

GPU-accelerated social-scientific network visualization, built entirely in pure Rust.

## Overview

Lucid Visualization Suite (LVS) is a high-performance desktop application for
animated 3-D network visualization of social-scientific data.  It is a ground-up
Rust rewrite of the legacy ETV Suite, replacing Python + Java with:

| Component | Purpose |
|-----------|---------|
| **LV** (Lucid Visualization) | `wgpu`-based GPU-instanced renderer |
| **AS** (AlignSpace) | Distance-matrix embedding, structural equivalence, and Procrustes alignment |
| **MF** (MatrixForge) | Co-occurrence + PMI/NPPMI text-analysis pipeline |

## Sample data

The repository includes small, synthetic sample datasets for local test runs and
experimentation:

- `tests/fixtures/sample_adjacency.csv` - example weighted adjacency matrix for
  the network pipeline.
- `tests/fixtures/sample_corpus.txt` - example text corpus for the text-analysis
  pipeline.

These files are safe to publish and are intended to be downloaded, inspected,
and reused for smoke tests. Generated media under `demo_output/` is excluded
from version control; users can recreate outputs locally from the sample data.

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Rust + Cargo | 1.78 (stable) |
| `wgpu`-compatible GPU | Vulkan, Metal, or DX12 |

On **Linux** you additionally need the Vulkan loader and development headers:

```bash
# Debian / Ubuntu
sudo apt-get install libvulkan-dev vulkan-tools

# Fedora / RHEL
sudo dnf install vulkan-loader-devel
```

On **Windows** a DX12-capable GPU and driver are sufficient (no extra installs).

## Building

### Debug build (fast iteration)

```bash
cargo build -p lv-app
```

### Release build (optimised)

```bash
cargo build --release -p lv-app
```

### Run the application

```bash
cargo run --release -p lv-app
# or directly:
./target/release/lucid-viz
```

## Running tests

```bash
cargo test --workspace
```

## Running benchmarks

```bash
# MDS pipeline benchmarks
cargo bench -p as-pipeline

# Text-analysis pipeline benchmarks
cargo bench -p mf-pipeline

# Renderer / LIS buffer benchmarks
cargo bench -p lv-renderer
```

## Feature flags

### `lv-app`

| Flag | Default | Description |
|------|---------|-------------|
| `audio` | ✅ | MIDI audio engine via `lv-audio` |
| `midi` | ❌ | Adds live MIDI-device output (implies `audio`) |
| `export` | ✅ | PNG/video export via `lv-export` |
| `native-io` | ✅ | XLSX/JSON file I/O |
| `wasm` | ❌ | WASM32 compilation scaffold |

### `lv-data`

| Flag | Default | Description |
|------|---------|-------------|
| `native-io` | ✅ | Enables `read_etv_xlsx` and `load_dataset_json` |
| `wasm` | ❌ | Disables all file I/O for `wasm32-unknown-unknown` |

## WASM scaffold build (experimental)

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown \
            --no-default-features --features wasm \
            -p lv-data -p as-pipeline -p mf-pipeline
```

> **Note:** The WASM build is a scaffold only.  The full GPU renderer requires a
> WebGPU backend not yet wired in this release.

## Native installer

Use [`cargo-bundle`](https://github.com/burtonageo/cargo-bundle):

```bash
cargo install cargo-bundle
cargo bundle --release -p lv-app
```

This produces a `.app` bundle on macOS and a `.deb` / standalone binary on Linux.
On Windows, the `build.rs` script embeds version info and the application icon
via `winres`.

## Project structure

```
lucid-viz/
├── crates/
│   ├── lv-data/        XLSX/JSON I/O + shared data model
│   ├── as-pipeline/    AlignSpace: distance matrices, MDS, Procrustes, centrality
│   ├── mf-pipeline/    MatrixForge: text -> co-occurrence -> similarity matrices
│   ├── lv-renderer/    wgpu GPU renderer + LIS buffer
│   ├── lv-gui/         egui panels and workspace
│   ├── lv-audio/       MIDI audio engine
│   ├── lv-export/      PNG/video export
│   └── lv-app/         Main binary (integrates all crates)
├── tests/integration/  Integration test suite
├── assets/
│   ├── shaders/        WGSL shader sources
│   └── icons/          Application icons
└── .github/workflows/  CI configuration
```

## MF -> AS bridge

- `mf-pipeline` produces similarity matrices (`similarity_matrix`, `nppmi_matrix`, `ppmi_matrix`) plus MF-side centrality.
- `as-pipeline` converts MF similarity into a distance matrix through `mf_output_to_distance_matrix` before running MDS (`mf_output_to_se_matrix` remains as a compatibility wrapper).
- `MdsDimMode::Visual` is the legacy public enum name for the current 2D planar layout mode.
- `as-pipeline` can also start from adjacency matrices, where it computes structural-equivalence distances internally.
- Precomputed distance inputs do not imply a graph, so AS now marks centrality as unavailable for those runs instead of emitting zero-filled metrics.
- Temporal AS runs now support `NormalizationMode::Independent` (per-slice scaling) and `NormalizationMode::Global` (one shared scale across the whole series).
- Temporal Procrustes now supports both chained `TimeSeries` alignment and `TimeSeriesAnchored`, which keeps every later slice aligned to slice 0 to reduce long-run drift.
- LV dataset loading canonicalizes `all_labels` from sheet rows and validates the result before runtime use.
- Dataset -> AS ingestion, AS export, LV JSON/XLSX I/O, and LV ego/runtime overlays now preserve directed `EdgeRow { from, to }` semantics end-to-end.
- AS centrality remains intentionally undirected today for backward compatibility: it derives metrics from the upper triangle of the adjacency matrix even when the preserved LV edge set is directed.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
