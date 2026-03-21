//! Full render cycle integration test — Phase 4 + Phase 9.
//!
//! Runs a headless wgpu render of a single LIS frame and verifies:
//!  1. The returned pixel buffer is the right size.
//!  2. The image is not entirely black (objects were rendered).
//!  3. Phase 9: multi-sheet LIS buffer frame count assertion.
//!  4. Phase 9: SHA-256 golden-hash assertions across 5 key frames.

use lv_data::{EtvDataset, EtvRow, EtvSheet, LisConfig, ShapeKind};
use lv_renderer::{build_lis_buffer, render_headless};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

fn make_small_dataset() -> EtvDataset {
    let rows: Vec<EtvRow> = (0..10)
        .map(|i| EtvRow {
            label: format!("n{i}"),
            x: (i as f64 - 5.0) * 50.0,
            y: 0.0,
            z: 0.0,
            size: 10.0,
            size_alpha: 0.5,
            spin_x: 0.0,
            spin_y: 0.0,
            spin_z: 0.0,
            shape: ShapeKind::ALL[i % ShapeKind::ALL.len()],
            color_r: 0.8,
            color_g: 0.4,
            color_b: 0.2,
            note: 60,
            instrument: 0,
            channel: 0,
            velocity: 100,
            cluster_value: 0.0,
            beats: 1,
        })
        .collect();

    let all_labels = rows.iter().map(|r| r.label.clone()).collect();
    EtvDataset {
        source_path: None,
        sheets: vec![EtvSheet {
            name: "T0".into(),
            sheet_index: 0,
            rows,
            edges: vec![],
        }],
        all_labels,
    }
}

/// Build a synthetic "canadian_migration_small"-equivalent dataset:
/// 3 sheets × 5 labelled objects (positions vary per sheet).
fn make_migration_dataset() -> EtvDataset {
    let labels: Vec<String> = (0..5).map(|i| format!("city_{i}")).collect();
    let x_base = [-200.0, -100.0, 0.0, 100.0, 200.0f64];
    let y_base = [50.0, -30.0, 80.0, -60.0, 10.0f64];

    let sheets: Vec<EtvSheet> = (0..3)
        .map(|si| {
            let rows: Vec<EtvRow> = (0..5)
                .map(|i| {
                    let shift = si as f64 * 20.0;
                    EtvRow {
                        label: labels[i].clone(),
                        x: x_base[i] + shift,
                        y: y_base[i] - shift * 0.5,
                        z: 0.0,
                        size: 10.0,
                        size_alpha: 1.0,
                        spin_x: 0.0,
                        spin_y: 0.0,
                        spin_z: 0.0,
                        shape: ShapeKind::Sphere,
                        color_r: (0.3 + 0.1 * si as f64) as f32,
                        color_g: 0.6,
                        color_b: (0.8 - 0.1 * si as f64) as f32,
                        note: 60,
                        instrument: 0,
                        channel: 0,
                        velocity: 80,
                        cluster_value: i as f64,
                        beats: 1,
                    }
                })
                .collect();
            EtvSheet {
                name: format!("Year{}", 2010 + si * 5),
                sheet_index: si,
                rows,
                edges: Vec::new(),
            }
        })
        .collect();

    EtvDataset {
        source_path: None,
        sheets,
        all_labels: labels,
    }
}

/// Compute hex-encoded SHA-256 of a raw pixel buffer.
fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

/// Load or create the golden-hashes store.
/// Keys are strings like `"frame_{idx}"`.
fn load_golden(path: &Path) -> HashMap<String, String> {
    if path.exists() {
        let s = std::fs::read_to_string(path).unwrap_or_default();
        toml::from_str(&s).unwrap_or_default()
    } else {
        HashMap::new()
    }
}

/// Write the golden-hashes store atomically.
fn save_golden(path: &Path, hashes: &HashMap<String, String>) {
    let s = toml::to_string(hashes).expect("serialise golden hashes");
    std::fs::write(path, s).expect("write golden_hashes.toml");
}

#[test]
fn headless_render_produces_non_black_image() {
    let dataset = make_small_dataset();
    let config = LisConfig::default();
    let lis_buffer = build_lis_buffer(&dataset, &config);

    assert!(
        !lis_buffer.frames.is_empty(),
        "LIS buffer should have at least one frame"
    );

    let frame = &lis_buffer.frames[0];
    let (width, height) = (256u32, 256u32);
    let pixels = render_headless(frame, width, height);

    // Correct byte count
    assert_eq!(
        pixels.len(),
        (width * height * 4) as usize,
        "pixel buffer size mismatch"
    );

    // Not entirely black — at least one non-background pixel
    let has_non_background = pixels.chunks(4).any(|px| {
        // Background is (13, 13, 20, 255) ≈ the clear color; anything brighter counts
        px[0] > 30 || px[1] > 30 || px[2] > 30
    });
    assert!(
        has_non_background,
        "rendered image is entirely black — nothing was drawn"
    );
}

#[test]
fn lis_buffer_has_correct_frame_count() {
    let dataset = make_small_dataset();
    let config = LisConfig {
        lis_value: 10,
        ..Default::default()
    };
    let buf = build_lis_buffer(&dataset, &config);

    // 1 sheet -> renderer keeps a static animation segment of `lis` frames.
    assert_eq!(buf.total_frames, 10);
    assert_eq!(buf.frames.len(), 10);
    assert!(!buf.streaming);
}

/// Phase 9 — multi-sheet LIS buffer frame count assertion.
/// 3 sheets -> 2 transitions; 2 * LIS=30 = 60 frames.
#[test]
fn migration_dataset_lis_buffer_60_frames() {
    let dataset = make_migration_dataset();
    let config = LisConfig {
        lis_value: 30,
        ..Default::default()
    };
    let buf = build_lis_buffer(&dataset, &config);

    assert_eq!(
        buf.total_frames, 60,
        "Expected 60 total frames for 3-sheet dataset with LIS=30"
    );
    assert_eq!(buf.frames.len(), 60);
}

/// Phase 9 — golden-hash assertions across 5 key frames.
///
/// First run: writes `tests/integration/golden_hashes.toml`.
/// Subsequent runs: asserts hashes match.
#[test]
fn migration_dataset_golden_hashes() {
    let dataset = make_migration_dataset();
    let config = LisConfig {
        lis_value: 30,
        ..Default::default()
    };
    let buf = build_lis_buffer(&dataset, &config);

    assert_eq!(buf.total_frames, 60);

    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // crates/lv-renderer -> crates
        .and_then(|p| p.parent()) // crates -> lucid-viz workspace root
        .map(|p| {
            p.join("tests")
                .join("integration")
                .join("golden_hashes.toml")
        })
        .expect("could not compute golden_hashes.toml path");

    let mut stored = load_golden(&golden_path);
    let (width, height) = (256u32, 256u32);

    let key_frames = [0usize, 15, 30, 45, 59];
    let mut computed: HashMap<String, String> = HashMap::new();

    for &idx in &key_frames {
        let frame = &buf.frames[idx];
        let pixels = render_headless(frame, width, height);
        assert_eq!(
            pixels.len(),
            (width * height * 4) as usize,
            "pixel buffer size mismatch at frame {idx}"
        );
        let hash = sha256_hex(&pixels);
        computed.insert(format!("frame_{idx}"), hash);
    }

    if stored.is_empty() {
        // First run: seed the golden file.
        for (k, v) in &computed {
            stored.insert(k.clone(), v.clone());
        }
        save_golden(&golden_path, &stored);
        println!("Golden hashes written to {golden_path:?}");
    } else {
        // Subsequent runs: assert hashes match.
        for &idx in &key_frames {
            let key = format!("frame_{idx}");
            let expected = stored
                .get(&key)
                .unwrap_or_else(|| panic!("Missing golden hash for {key}"));
            let actual = computed.get(&key).unwrap();
            assert_eq!(
                actual, expected,
                "Golden hash mismatch at {key}: expected {expected}, got {actual}"
            );
        }
    }
}
