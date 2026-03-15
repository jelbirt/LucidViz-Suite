//! Axis-aligned box mesh.
//!
//! The `size_alpha` instance field is interpreted as the depth/width ratio so
//! the shader scales depth accordingly; however the mesh itself is a unit cube
//! centred at the origin. Scaling happens per-instance on the GPU.

use super::Vertex;

pub fn build(_lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    // 6 faces × 4 vertices each = 24 unique vertices (each face has its own
    // normal so we cannot share corners).
    #[rustfmt::skip]
    let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
        // normal,          four corners (CCW when viewed from outside)
        ([0.0,  1.0,  0.0], [[-0.5,0.5,-0.5],[ 0.5,0.5,-0.5],[ 0.5,0.5, 0.5],[-0.5,0.5, 0.5]]), // +Y top
        ([0.0, -1.0,  0.0], [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5]]), // -Y bottom
        ([0.0,  0.0,  1.0], [[-0.5,-0.5, 0.5],[ 0.5,-0.5, 0.5],[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]]), // +Z front
        ([0.0,  0.0, -1.0], [[ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[ 0.5, 0.5,-0.5]]), // -Z back
        ([1.0,  0.0,  0.0], [[ 0.5,-0.5, 0.5],[ 0.5,-0.5,-0.5],[ 0.5, 0.5,-0.5],[ 0.5, 0.5, 0.5]]), // +X right
        ([-1.0, 0.0,  0.0], [[-0.5,-0.5,-0.5],[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5]]), // -X left
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for &pos in corners {
            vertices.push(Vertex {
                position: pos,
                normal: *normal,
            });
        }
        // Two triangles per face (quad = 0,1,2 + 0,2,3)
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (vertices, indices)
}
