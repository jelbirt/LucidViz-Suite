//! Built-in demo dataset: Global Trade Network, 2017-2024.
//!
//! 48 country-nodes in 5 regional clusters. Each of the 8 sheets represents
//! one calendar year. Node positions drift over time to show structural change
//! (e.g. US-China decoupling, post-Brexit UK drift, emerging-market rise).
//!
//! Encoding:
//!   Shape   -> region  (Sphere=Americas, Cube=Europe, Cylinder=Asia-Pacific,
//!                        Torus=Africa/MENA, Pyramid=Central/South Asia)
//!   Color   -> region palette (saturated, distinct)
//!   Size    -> GDP tier (0.18 - 0.55)
//!   Cluster -> GDP tier 0-4
//!   Spin    -> trade velocity (large economies spin faster on Y-axis)
//!   Edges   -> top bilateral trade relationships, strength ~ trade volume

use lv_data::{EtvDataset, EtvRow, EtvSheet, ShapeKind};

pub fn make_demo_dataset() -> EtvDataset {
    // ── node catalogue ────────────────────────────────────────────────────────
    // (label, region, gdp_tier 0-4, base_x, base_y, base_z, spin_y deg/frame)
    #[allow(clippy::type_complexity)]
    let nodes: &[(&str, usize, u32, f64, f64, f64, f64)] = &[
        // Americas  (region 0)  shape=Sphere  colour=blue
        ("USA", 0, 4, 4.0, 1.5, 0.5, 0.8),
        ("China", 2, 4, -4.0, 1.0, 0.0, 0.8),
        ("Canada", 0, 2, 3.5, 3.0, 1.0, 0.3),
        ("Mexico", 0, 2, 3.0, -0.5, -0.5, 0.3),
        ("Brazil", 0, 3, 2.0, -2.5, 1.5, 0.5),
        ("Argentina", 0, 1, 1.5, -3.5, 0.5, 0.2),
        ("Colombia", 0, 1, 2.5, -2.0, -1.0, 0.2),
        ("Chile", 0, 1, 1.0, -3.0, 2.0, 0.2),
        ("Peru", 0, 0, 1.5, -2.8, 2.5, 0.1),
        ("Venezuela", 0, 0, 2.0, -1.5, -2.0, 0.1),
        // Europe  (region 1)  shape=Cube  colour=green
        ("Germany", 1, 3, -1.0, 3.5, -1.5, 0.6),
        ("France", 1, 3, -0.5, 3.0, -2.5, 0.5),
        ("UK", 1, 3, 0.5, 3.5, -3.0, 0.5),
        ("Italy", 1, 2, 0.0, 2.5, -2.0, 0.4),
        ("Spain", 1, 2, -0.5, 2.0, -3.0, 0.3),
        ("Netherlands", 1, 2, -1.5, 3.8, -0.5, 0.4),
        ("Belgium", 1, 1, -1.8, 3.2, -1.0, 0.2),
        ("Sweden", 1, 1, -0.8, 4.5, -1.0, 0.2),
        ("Poland", 1, 1, 0.2, 4.0, -0.5, 0.2),
        ("Switzerland", 1, 2, -0.3, 3.0, -1.5, 0.3),
        // Asia-Pacific  (region 2)  shape=Cylinder  colour=red
        ("Japan", 2, 3, -3.0, 2.0, 2.0, 0.6),
        ("South Korea", 2, 2, -3.5, 1.5, 3.0, 0.5),
        ("India", 2, 3, -2.0, 0.0, 3.5, 0.6),
        ("Australia", 2, 2, -2.5, -2.0, 4.0, 0.3),
        ("Indonesia", 2, 2, -3.5, -1.0, 4.5, 0.3),
        ("Thailand", 2, 1, -4.0, -0.5, 3.5, 0.2),
        ("Vietnam", 2, 1, -4.5, 0.0, 3.0, 0.2),
        ("Malaysia", 2, 1, -4.0, -1.5, 3.0, 0.2),
        ("Philippines", 2, 1, -3.8, -2.0, 2.5, 0.2),
        ("New Zealand", 2, 0, -2.0, -3.0, 4.5, 0.1),
        // Africa / MENA  (region 3)  shape=Torus  colour=orange
        ("Saudi Arabia", 3, 3, 0.5, 0.0, -4.0, 0.5),
        ("UAE", 3, 2, 1.0, 0.5, -4.5, 0.4),
        ("Turkey", 3, 2, 0.5, 1.5, -3.5, 0.3),
        ("South Africa", 3, 1, -1.0, -1.5, -3.5, 0.2),
        ("Nigeria", 3, 1, -0.5, -2.0, -4.0, 0.2),
        ("Egypt", 3, 1, 0.0, -0.5, -4.5, 0.2),
        ("Morocco", 3, 0, -0.5, 0.0, -5.0, 0.1),
        ("Kenya", 3, 0, -1.5, -2.5, -3.0, 0.1),
        ("Ethiopia", 3, 0, -1.0, -3.0, -3.5, 0.1),
        ("Ghana", 3, 0, 0.0, -2.8, -4.5, 0.1),
        // Central / South Asia  (region 4)  shape=Pyramid  colour=purple
        ("Russia", 4, 3, -2.0, 2.5, -1.0, 0.5),
        ("Kazakhstan", 4, 1, -2.5, 1.5, -2.0, 0.2),
        ("Pakistan", 4, 1, -1.5, 0.5, 2.5, 0.2),
        ("Bangladesh", 4, 0, -1.0, 0.0, 4.0, 0.1),
        ("Sri Lanka", 4, 0, -1.5, -0.5, 4.5, 0.1),
        ("Uzbekistan", 4, 0, -2.0, 1.0, -1.5, 0.1),
        ("Iran", 4, 1, -0.5, 0.5, -3.0, 0.2),
        ("Iraq", 4, 0, 0.0, 0.0, -3.5, 0.1),
    ];

    // ── region visual encoding ────────────────────────────────────────────────
    let region_shape = [
        ShapeKind::Sphere,   // Americas
        ShapeKind::Cube,     // Europe
        ShapeKind::Cylinder, // Asia-Pacific
        ShapeKind::Torus,    // Africa/MENA
        ShapeKind::Pyramid,  // Central/South Asia
    ];
    let region_color: [(f32, f32, f32); 5] = [
        (0.25, 0.55, 0.95), // blue   - Americas
        (0.25, 0.80, 0.45), // green  - Europe
        (0.95, 0.30, 0.30), // red    - Asia-Pacific
        (0.95, 0.65, 0.15), // orange - Africa/MENA
        (0.75, 0.35, 0.95), // purple - Central/South Asia
    ];
    let tier_size = [0.18_f64, 0.25, 0.33, 0.44, 0.55];

    // ── top bilateral trade edges ──
    let edge_defs: &[(&str, &str, f64, f64)] = &[
        ("USA", "Canada", 0.90, 0.01),
        ("USA", "Mexico", 0.85, 0.01),
        ("USA", "China", 0.80, -0.04),
        ("USA", "Japan", 0.70, 0.00),
        ("USA", "Germany", 0.65, 0.00),
        ("USA", "UK", 0.60, 0.01),
        ("China", "Japan", 0.75, -0.02),
        ("China", "South Korea", 0.78, -0.01),
        ("China", "Germany", 0.70, 0.00),
        ("China", "Australia", 0.65, -0.03),
        ("China", "Vietnam", 0.55, 0.05),
        ("China", "India", 0.50, -0.03),
        ("Germany", "France", 0.88, 0.00),
        ("Germany", "Netherlands", 0.80, 0.00),
        ("Germany", "Italy", 0.75, 0.00),
        ("Germany", "UK", 0.72, -0.02),
        ("Germany", "Poland", 0.60, 0.03),
        ("Germany", "Russia", 0.55, -0.10),
        ("UK", "France", 0.68, -0.02),
        ("UK", "Netherlands", 0.65, -0.01),
        ("Japan", "South Korea", 0.72, 0.00),
        ("Japan", "Australia", 0.65, 0.01),
        ("Japan", "India", 0.50, 0.03),
        ("India", "UAE", 0.60, 0.04),
        ("India", "Saudi Arabia", 0.55, 0.02),
        ("India", "USA", 0.58, 0.03),
        ("Brazil", "China", 0.65, 0.04),
        ("Brazil", "USA", 0.55, 0.00),
        ("Russia", "China", 0.70, 0.08),
        ("Russia", "India", 0.45, 0.06),
        ("Saudi Arabia", "China", 0.68, 0.05),
        ("Saudi Arabia", "India", 0.62, 0.03),
        ("UAE", "India", 0.58, 0.04),
        ("Vietnam", "USA", 0.52, 0.06),
        ("Vietnam", "South Korea", 0.55, 0.04),
        ("South Korea", "USA", 0.65, 0.00),
        ("Australia", "USA", 0.58, 0.01),
        ("Indonesia", "China", 0.55, 0.03),
        ("Mexico", "Canada", 0.65, 0.02),
        ("Turkey", "Germany", 0.60, 0.01),
    ];

    // ── time-series drift vectors ──
    let drift: std::collections::HashMap<&str, (f64, f64, f64)> = [
        ("UK", (-0.15, -0.05, 0.08)),
        ("Russia", (-0.35, 0.00, 0.30)),
        ("Vietnam", (0.25, 0.05, -0.15)),
        ("Indonesia", (0.20, 0.03, -0.10)),
        ("India", (-0.10, 0.05, 0.05)),
        ("Germany", (0.00, 0.00, 0.05)),
        ("China", (0.10, 0.05, -0.05)),
        ("USA", (-0.05, -0.02, 0.02)),
        ("Saudi Arabia", (0.05, 0.05, 0.10)),
        ("UAE", (0.05, 0.05, 0.10)),
        ("Brazil", (0.05, 0.03, 0.00)),
        ("Poland", (-0.10, 0.00, -0.05)),
        ("Kazakhstan", (-0.10, 0.00, 0.15)),
    ]
    .iter()
    .cloned()
    .collect();

    let num_sheets: usize = 8;
    let all_labels: Vec<String> = nodes.iter().map(|(l, ..)| l.to_string()).collect();

    let sheets: Vec<EtvSheet> = (0..num_sheets)
        .map(|s| {
            let year_offset = s as f64;

            let rows: Vec<EtvRow> = nodes
                .iter()
                .map(|(label, region, gdp_tier, bx, by, bz, spin_y)| {
                    let (dx, dy, dz) = drift.get(label).copied().unwrap_or((0.0, 0.0, 0.0));

                    let h = label
                        .bytes()
                        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
                    let noise_scale = 0.08;
                    let nx = noise_scale
                        * ((h.wrapping_mul(1_000_003).wrapping_add(s as u64 * 7)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;
                    let ny = noise_scale
                        * ((h.wrapping_mul(1_000_033).wrapping_add(s as u64 * 13)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;
                    let nz = noise_scale
                        * ((h.wrapping_mul(1_000_099).wrapping_add(s as u64 * 19)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;

                    let (cr, cg, cb) = region_color[*region];
                    EtvRow {
                        label: label.to_string(),
                        x: bx + dx * year_offset + nx,
                        y: by + dy * year_offset + ny,
                        z: bz + dz * year_offset + nz,
                        size: tier_size[*gdp_tier as usize],
                        shape: region_shape[*region],
                        color_r: cr,
                        color_g: cg,
                        color_b: cb,
                        cluster_value: *gdp_tier as f64,
                        spin_y: *spin_y,
                        ..EtvRow::default()
                    }
                })
                .collect();

            let edges: Vec<lv_data::EdgeRow> = edge_defs
                .iter()
                .filter_map(|(from, to, base, growth)| {
                    let strength = (base + growth * year_offset).clamp(0.0, 1.0);
                    if strength > 0.1 {
                        Some(lv_data::EdgeRow {
                            from: from.to_string(),
                            to: to.to_string(),
                            strength,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            EtvSheet {
                name: format!("{}", 2017 + s),
                sheet_index: s,
                rows,
                edges,
            }
        })
        .collect();

    EtvDataset {
        source_path: None,
        sheets,
        all_labels,
    }
}
