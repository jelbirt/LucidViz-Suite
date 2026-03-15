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
