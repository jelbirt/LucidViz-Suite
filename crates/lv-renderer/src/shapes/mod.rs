//! GPU mesh generators for every `ShapeKind`.

pub mod cube;
pub mod cylinder;
pub mod point;
pub mod pyramid;
pub mod sphere;
pub mod torus;

use wgpu::util::DeviceExt;

/// A single mesh vertex: position + surface normal (both in object space).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
    };
}

/// An uploaded mesh ready for rendering.
pub struct GpuMesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh {
    /// Upload a `(vertices, indices)` pair to the GPU.
    pub fn upload(
        device: &wgpu::Device,
        label: &str,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}_vb")),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label}_ib")),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

/// LOD level selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lod {
    Low = 0,
    Mid = 1,
    High = 2,
}

impl Lod {
    pub fn from_index(i: usize) -> Self {
        match i {
            1 => Lod::Mid,
            2 => Lod::High,
            _ => Lod::Low,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate that a mesh has sensible vertex/index counts and finite normals.
    fn validate_mesh(name: &str, vertices: &[Vertex], indices: &[u32]) {
        assert!(
            !vertices.is_empty(),
            "{name}: mesh must have at least one vertex"
        );
        assert!(
            !indices.is_empty(),
            "{name}: mesh must have at least one index"
        );
        // Index count must be a multiple of 3 (triangles).
        assert_eq!(
            indices.len() % 3,
            0,
            "{name}: index count {} is not a multiple of 3",
            indices.len()
        );
        // All indices must be in range.
        let max_idx = *indices.iter().max().unwrap() as usize;
        assert!(
            max_idx < vertices.len(),
            "{name}: index {max_idx} out of range (vertex count {})",
            vertices.len()
        );
        // All positions and normals must be finite.
        for (i, v) in vertices.iter().enumerate() {
            for c in 0..3 {
                assert!(
                    v.position[c].is_finite(),
                    "{name}: vertex {i} position[{c}] is not finite"
                );
                assert!(
                    v.normal[c].is_finite(),
                    "{name}: vertex {i} normal[{c}] is not finite"
                );
            }
            // Normal should be unit-length (tolerance for discretization).
            let len = (v.normal[0].powi(2) + v.normal[1].powi(2) + v.normal[2].powi(2)).sqrt();
            assert!(
                (len - 1.0).abs() < 0.05,
                "{name}: vertex {i} normal length {len} deviates from 1.0"
            );
        }
    }

    #[test]
    fn cube_mesh_valid() {
        let (v, i) = cube::build(Lod::Low);
        validate_mesh("cube", &v, &i);
        assert_eq!(v.len(), 24, "cube: 6 faces * 4 vertices = 24");
        assert_eq!(i.len(), 36, "cube: 6 faces * 6 indices = 36");
    }

    #[test]
    fn sphere_mesh_valid_all_lods() {
        for lod in [Lod::Low, Lod::Mid, Lod::High] {
            let (v, i) = sphere::build(lod);
            validate_mesh(&format!("sphere/{lod:?}"), &v, &i);
            assert!(v.len() > 20, "sphere should have many vertices");
        }
    }

    #[test]
    fn sphere_lod_increases_detail() {
        let (v_low, _) = sphere::build(Lod::Low);
        let (v_mid, _) = sphere::build(Lod::Mid);
        let (v_high, _) = sphere::build(Lod::High);
        assert!(v_mid.len() > v_low.len(), "mid > low vertex count");
        assert!(v_high.len() > v_mid.len(), "high > mid vertex count");
    }

    #[test]
    fn cylinder_mesh_valid_all_lods() {
        for lod in [Lod::Low, Lod::Mid, Lod::High] {
            let (v, i) = cylinder::build(lod);
            validate_mesh(&format!("cylinder/{lod:?}"), &v, &i);
        }
    }

    #[test]
    fn pyramid_mesh_valid() {
        let (v, i) = pyramid::build(Lod::Low);
        validate_mesh("pyramid", &v, &i);
        // 4 side triangles * 3 verts + 4 base verts = 16
        assert_eq!(v.len(), 16, "pyramid: 4*3 + 4 = 16 vertices");
        // 4 side triangles * 3 + 2 base triangles * 3 = 18
        assert_eq!(i.len(), 18, "pyramid: 12 + 6 = 18 indices");
    }

    #[test]
    fn torus_mesh_valid_all_lods() {
        for lod in [Lod::Low, Lod::Mid, Lod::High] {
            let (v, i) = torus::build(lod);
            validate_mesh(&format!("torus/{lod:?}"), &v, &i);
        }
    }

    #[test]
    fn point_mesh_equals_low_sphere() {
        let (v_point, i_point) = point::build(Lod::Low);
        let (v_sphere, i_sphere) = sphere::build(Lod::Low);
        assert_eq!(v_point.len(), v_sphere.len());
        assert_eq!(i_point.len(), i_sphere.len());
    }

    #[test]
    fn lod_from_index_mapping() {
        assert_eq!(Lod::from_index(0), Lod::Low);
        assert_eq!(Lod::from_index(1), Lod::Mid);
        assert_eq!(Lod::from_index(2), Lod::High);
        assert_eq!(Lod::from_index(99), Lod::Low); // fallback
    }
}
