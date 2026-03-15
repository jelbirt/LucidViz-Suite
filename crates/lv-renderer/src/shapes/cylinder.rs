//! Capped cylinder mesh.
//!
//! `size_alpha` is interpreted as the height/radius ratio by the caller.
//! The mesh is a unit-radius, unit-height cylinder centred at the origin.
//!
//! LOD selects the number of radial segments:
//!  - Low:  16
//!  - Mid:  24
//!  - High: 48

use super::Vertex;
use std::f32::consts::TAU;

pub fn build(lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    let segs: u32 = match lod {
        super::Lod::Low => 16,
        super::Lod::Mid => 24,
        super::Lod::High => 48,
    };

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let half = 0.5_f32;

    // ── Side wall ────────────────────────────────────────────────────────────
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let ang = t * TAU;
        let nx = ang.cos();
        let nz = ang.sin();

        vertices.push(Vertex {
            position: [nx, -half, nz],
            normal: [nx, 0.0, nz],
        });
        vertices.push(Vertex {
            position: [nx, half, nz],
            normal: [nx, 0.0, nz],
        });
    }

    for i in 0..segs {
        let b = i * 2;
        indices.extend_from_slice(&[b, b + 1, b + 2, b + 1, b + 3, b + 2]);
    }

    // ── Top cap ──────────────────────────────────────────────────────────────
    let top_centre = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, half, 0.0],
        normal: [0.0, 1.0, 0.0],
    });

    let top_ring_start = vertices.len() as u32;
    for i in 0..segs {
        let ang = i as f32 / segs as f32 * TAU;
        vertices.push(Vertex {
            position: [ang.cos(), half, ang.sin()],
            normal: [0.0, 1.0, 0.0],
        });
    }
    for i in 0..segs {
        let a = top_ring_start + i;
        let b = top_ring_start + (i + 1) % segs;
        indices.extend_from_slice(&[top_centre, a, b]);
    }

    // ── Bottom cap ───────────────────────────────────────────────────────────
    let bot_centre = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, -half, 0.0],
        normal: [0.0, -1.0, 0.0],
    });

    let bot_ring_start = vertices.len() as u32;
    for i in 0..segs {
        let ang = i as f32 / segs as f32 * TAU;
        vertices.push(Vertex {
            position: [ang.cos(), -half, ang.sin()],
            normal: [0.0, -1.0, 0.0],
        });
    }
    for i in 0..segs {
        let a = bot_ring_start + i;
        let b = bot_ring_start + (i + 1) % segs;
        // Winding reversed for bottom face
        indices.extend_from_slice(&[bot_centre, b, a]);
    }

    (vertices, indices)
}
