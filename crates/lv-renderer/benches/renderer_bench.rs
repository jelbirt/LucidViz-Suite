use criterion::{criterion_group, criterion_main, Criterion};
use lv_data::{LisConfig, LvDataset, LvRow, LvSheet, ShapeKind};
use lv_renderer::lis::{build_lis_buffer, compute_frame};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;

// ── helpers ────────────────────────────────────────────────────────────────────

fn fake_dataset(n_objects: usize, n_sheets: usize) -> LvDataset {
    let mut rng = SmallRng::seed_from_u64(3);
    let sheets: Vec<LvSheet> = (0..n_sheets)
        .map(|si| {
            let rows: Vec<LvRow> = (0..n_objects)
                .map(|i| LvRow {
                    label: format!("obj_{i}"),
                    x: rng.random_range(-1.0..1.0),
                    y: rng.random_range(-1.0..1.0),
                    z: rng.random_range(-1.0..1.0),
                    size: rng.random_range(0.1..1.0),
                    size_alpha: 1.0,
                    spin_x: 0.0,
                    spin_y: 0.0,
                    spin_z: 0.0,
                    shape: ShapeKind::Sphere,
                    color_r: rng.random_range(0.0..1.0),
                    color_g: rng.random_range(0.0..1.0),
                    color_b: rng.random_range(0.0..1.0),
                    note: 0,
                    instrument: 0,
                    channel: 0,
                    velocity: 64,
                    cluster_value: rng.random_range(0.0..5.0),
                    beats: 1,
                })
                .collect();
            LvSheet {
                name: format!("Sheet{si}"),
                sheet_index: si,
                rows,
                edges: Vec::new(),
            }
        })
        .collect();
    LvDataset {
        source_path: None,
        sheets,
        all_labels: (0..n_objects).map(|i| format!("obj_{i}")).collect(),
    }
}

// ── renderer benchmarks ───────────────────────────────────────────────────────

fn bench_lis_buffer_build(c: &mut Criterion) {
    let ds = fake_dataset(100, 10);
    let cfg = LisConfig {
        lis_value: 60,
        ..LisConfig::default()
    };
    c.bench_function("lis_buffer_build_100obj_10sheets_lis60", |b| {
        b.iter(|| {
            let _ = build_lis_buffer(black_box(&ds), black_box(&cfg));
        })
    });
}

fn bench_compute_frame_10k(c: &mut Criterion) {
    let ds = fake_dataset(100, 100);
    let cfg = LisConfig {
        lis_value: 60,
        ..LisConfig::default()
    };
    c.bench_function("compute_frame_100obj_100sheets", |b| {
        b.iter(|| {
            let _ = compute_frame(black_box(&ds), black_box(&cfg), 0);
        })
    });
}

criterion_group!(
    renderer_benches,
    bench_lis_buffer_build,
    bench_compute_frame_10k
);
criterion_main!(renderer_benches);
