use as_pipeline::{
    mds::{classical::classical_mds, pivot::pivot_mds, smacof::smacof},
    types::{SeMatrix, SmacofConfig, SmacofInit},
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ── helpers ────────────────────────────────────────────────────────────────────

/// Generate a random symmetric n×n distance matrix.
fn random_dist(n: usize, seed: u64) -> SeMatrix {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let v: f64 = rng.gen_range(0.1..10.0);
            data[i * n + j] = v;
            data[j * n + i] = v;
        }
    }
    let labels: Vec<String> = (0..n).map(|i| format!("node_{i}")).collect();
    SeMatrix::new(labels, data)
}

// ── as-pipeline benchmarks ────────────────────────────────────────────────────

fn bench_se_n100(c: &mut Criterion) {
    let n = 100;
    let mut data = vec![0.0f64; n * n];
    let mut rng = SmallRng::seed_from_u64(1);
    for i in 0..n {
        for j in (i + 1)..n {
            let v: f64 = rng.gen_range(0.1..10.0);
            data[i * n + j] = v;
            data[j * n + i] = v;
        }
    }
    let labels: Vec<String> = (0..n).map(|i| format!("node_{i}")).collect();
    c.bench_function("se_matrix_new_n100", |b| {
        b.iter(|| {
            let _ = SeMatrix::new(black_box(labels.clone()), black_box(data.clone()));
        })
    });
}

fn bench_se_n1000(c: &mut Criterion) {
    let n = 1000;
    let mut data = vec![0.0f64; n * n];
    let mut rng = SmallRng::seed_from_u64(2);
    for i in 0..n {
        for j in (i + 1)..n {
            let v: f64 = rng.gen_range(0.1..10.0);
            data[i * n + j] = v;
            data[j * n + i] = v;
        }
    }
    let labels: Vec<String> = (0..n).map(|i| format!("node_{i}")).collect();
    c.bench_function("se_matrix_new_n1000", |b| {
        b.iter(|| {
            let _ = SeMatrix::new(black_box(labels.clone()), black_box(data.clone()));
        })
    });
}

fn bench_classical_mds(c: &mut Criterion) {
    let mut group = c.benchmark_group("classical_mds");
    for n in [100usize, 1000] {
        let dist = random_dist(n, n as u64);
        group.bench_with_input(BenchmarkId::from_parameter(n), &dist, |b, d| {
            b.iter(|| {
                let _ = classical_mds(black_box(d), 3).unwrap();
            })
        });
    }
    group.finish();
}

fn bench_smacof_n500(c: &mut Criterion) {
    let dist = random_dist(500, 42);
    let cfg = SmacofConfig {
        max_iter: 50,
        tolerance: 1e-4,
        init: SmacofInit::Random(99),
    };
    c.bench_function("smacof_n500_max50", |b| {
        b.iter(|| {
            let _ = smacof(black_box(&dist), 3, black_box(&cfg)).unwrap();
        })
    });
}

fn bench_pivot_mds_n5000(c: &mut Criterion) {
    let dist = random_dist(5000, 7);
    c.bench_function("pivot_mds_n5000_p50", |b| {
        b.iter(|| {
            let _ = pivot_mds(black_box(&dist), 3, 50).unwrap();
        })
    });
}

criterion_group!(
    mds_benches,
    bench_se_n100,
    bench_se_n1000,
    bench_classical_mds,
    bench_smacof_n500,
    bench_pivot_mds_n5000
);
criterion_main!(mds_benches);
