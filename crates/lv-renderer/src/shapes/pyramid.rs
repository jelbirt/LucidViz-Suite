//! Triangular pyramid (tetrahedron-like) mesh.
//!
//! `size_alpha` is the height / base-edge ratio.
//! The mesh is a square-base pyramid with apex at (0, 0.5, 0) and base at y=-0.5.

use super::Vertex;

pub fn build(_lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    // Four base corners and one apex.
    let apex = [0.0_f32, 0.5, 0.0];
    let bl = [-0.5_f32, -0.5, -0.5];
    let br = [0.5_f32, -0.5, -0.5];
    let fr = [0.5_f32, -0.5, 0.5];
    let fl = [-0.5_f32, -0.5, 0.5];

    let face_normal = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| -> [f32; 3] {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
        [n[0] / len, n[1] / len, n[2] / len]
    };

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // Helper: push a triangle with a flat face normal
    let mut push_tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
        let n = face_normal(a, b, c);
        let base = vertices.len() as u32;
        vertices.push(Vertex {
            position: a,
            normal: n,
        });
        vertices.push(Vertex {
            position: b,
            normal: n,
        });
        vertices.push(Vertex {
            position: c,
            normal: n,
        });
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    };

    // Four triangular side faces
    push_tri(apex, fl, bl);
    push_tri(apex, bl, br);
    push_tri(apex, br, fr);
    push_tri(apex, fr, fl);

    // Base (two triangles forming a quad)
    let base_n = [0.0_f32, -1.0, 0.0];
    let b0 = vertices.len() as u32;
    for &p in &[bl, br, fr, fl] {
        vertices.push(Vertex {
            position: p,
            normal: base_n,
        });
    }
    indices.extend_from_slice(&[b0, b0 + 2, b0 + 1, b0, b0 + 3, b0 + 2]);

    (vertices, indices)
}
