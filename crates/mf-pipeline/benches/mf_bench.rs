use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use mf_pipeline::{
    centrality::compute_betweenness_pg,
    cooccurrence::build_cooccurrence,
    types::{MfConfig, Token},
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ── helpers ────────────────────────────────────────────────────────────────────

fn random_tokens(n_tokens: usize, vocab_size: usize, seed: u64) -> Vec<Token> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_tokens)
        .map(|_| Token(format!("word_{}", rng.random_range(0..vocab_size))))
        .collect()
}

// ── mf-pipeline benchmarks ────────────────────────────────────────────────────

fn bench_cooccurrence_10k(c: &mut Criterion) {
    let tokens = random_tokens(10_000, 500, 1);
    let config = MfConfig {
        window_size: 5,
        slide_rate: 1,
        use_pmi: true,
        min_count: 2,
        min_pmi: 0.0,
        language: "en".to_string(),
        unicode_normalize: false,
        sim_to_dist: mf_pipeline::types::SimToDistMethod::Linear,
        ..Default::default()
    };
    c.bench_function("cooccurrence_10k_tokens", |b| {
        b.iter(|| {
            let _ = build_cooccurrence(black_box(&tokens), black_box(&config));
        })
    });
}

fn bench_brandes_n500(c: &mut Criterion) {
    // Build a random 500-node Erdős–Rényi graph (p=0.02 ≈ sparse).
    use petgraph::graph::UnGraph;
    let n = 500usize;
    let p = 0.02f64;
    let mut rng = SmallRng::seed_from_u64(42);
    let mut g: UnGraph<String, f64> = UnGraph::new_undirected();
    let nodes: Vec<_> = (0..n).map(|i| g.add_node(format!("n{i}"))).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.random::<f64>() < p {
                g.add_edge(nodes[i], nodes[j], 1.0);
            }
        }
    }
    c.bench_function("brandes_betweenness_n500", |b| {
        b.iter(|| {
            let _ = compute_betweenness_pg(black_box(&g));
        })
    });
}

criterion_group!(mf_benches, bench_cooccurrence_10k, bench_brandes_n500);
criterion_main!(mf_benches);
