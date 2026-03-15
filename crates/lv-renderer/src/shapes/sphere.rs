//! UV sphere mesh generator.
//!
//! LOD levels:
//!  - Low  (lod=0): 8 segments × 4 stacks
//!  - Mid  (lod=1): 16 × 8
//!  - High (lod=2): 32 × 16

use super::Vertex;

pub fn build(lod: super::Lod) -> (Vec<Vertex>, Vec<u32>) {
    let (segs, stacks) = match lod {
        super::Lod::Low => (8u32, 4u32),
        super::Lod::Mid => (16u32, 8u32),
        super::Lod::High => (32u32, 16u32),
    };

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices: rows from top (phi=0) to bottom (phi=PI)
    for j in 0..=stacks {
        let phi = std::f32::consts::PI * j as f32 / stacks as f32;
        let sp = phi.sin();
        let cp = phi.cos();

        for i in 0..=segs {
            let theta = 2.0 * std::f32::consts::PI * i as f32 / segs as f32;
            let st = theta.sin();
            let ct = theta.cos();

            let nx = sp * ct;
            let ny = cp;
            let nz = sp * st;

            vertices.push(Vertex {
                position: [nx, ny, nz],
                normal: [nx, ny, nz],
            });
        }
    }

    // Generate indices (two triangles per quad)
    for j in 0..stacks {
        for i in 0..segs {
            let row_cur = j * (segs + 1);
            let row_next = (j + 1) * (segs + 1);
            let a = row_cur + i;
            let b = row_cur + i + 1;
            let c = row_next + i;
            let d = row_next + i + 1;

            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    (vertices, indices)
}
