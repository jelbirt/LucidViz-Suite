//! Ring (torus) mesh.
//!
//! `size_alpha` is the tube-radius / major-radius ratio.
//! The mesh uses a fixed major radius of 0.5 so the torus fits within a unit
//! sphere. Tube radius is `0.5 * size_alpha` (caller scales via instance data).
//!
//! LOD selects segment counts (major × tube):
//!  - Low:  24 × 12
//!  - Mid:  32 × 16
//!  - High: 64 × 32

use super::Vertex;
use std::f32::consts::TAU;

pub fn build(lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    let (major_segs, tube_segs): (u32, u32) = match lod {
        super::Lod::Low => (24, 12),
        super::Lod::Mid => (32, 16),
        super::Lod::High => (64, 32),
    };

    let major_r = 0.5_f32;
    let tube_r = 0.15_f32; // default; caller overrides via size_alpha in shader

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for i in 0..=major_segs {
        let u = i as f32 / major_segs as f32 * TAU;
        let cu = u.cos();
        let su = u.sin();

        for j in 0..=tube_segs {
            let v = j as f32 / tube_segs as f32 * TAU;
            let cv = v.cos();
            let sv = v.sin();

            let x = (major_r + tube_r * cv) * cu;
            let y = tube_r * sv;
            let z = (major_r + tube_r * cv) * su;

            // Normal points away from the tube centre
            let nx = cv * cu;
            let ny = sv;
            let nz = cv * su;

            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx, ny, nz],
            });
        }
    }

    let ring = tube_segs + 1;
    for i in 0..major_segs {
        for j in 0..tube_segs {
            let a = i * ring + j;
            let b = (i + 1) * ring + j;
            let c = (i + 1) * ring + j + 1;
            let d = i * ring + j + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    (vertices, indices)
}
