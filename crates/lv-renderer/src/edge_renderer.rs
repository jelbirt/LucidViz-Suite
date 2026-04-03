//! `edge_renderer` — GPU line/edge renderer for the Ego Cluster system.
//!
//! Each logical edge is expanded to a camera-facing quad (4 vertices, 6 indices)
//! on the CPU before upload, matching the `edge.wgsl` vertex shader contract.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

// ── GpuEdge ───────────────────────────────────────────────────────────────────

/// One logical edge between two 3-D world-space positions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuEdge {
    pub from_pos: [f32; 3],
    pub to_pos: [f32; 3],
    pub color: [f32; 4],
}

// ── Vertex layout matching edge.wgsl ─────────────────────────────────────────
//
// edge.wgsl expects per-vertex (NOT instanced) attributes:
//   loc 0: from_pos  vec3<f32>   offset  0
//   loc 1: to_pos    vec3<f32>   offset 12
//   loc 2: color     vec4<f32>   offset 24
//   loc 3: side      f32         offset 40
// Total: 44 bytes per vertex
//
// Quad layout: 4 vertices per edge, 6 indices (2 triangles)
//   side values: +2.0, +1.0, -1.0, -2.0

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EdgeVertex {
    from_pos: [f32; 3],
    to_pos: [f32; 3],
    color: [f32; 4],
    side: f32,
}

static EDGE_VERTEX_ATTRS: &[wgpu::VertexAttribute] = &[
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 24,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32,
        offset: 40,
        shader_location: 3,
    },
];

pub const EDGE_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 44,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: EDGE_VERTEX_ATTRS,
};

// ── EdgeRenderer ──────────────────────────────────────────────────────────────

/// Maintains GPU buffers for all visible edges and draws them.
///
/// Buffers are reused across frames when capacity is sufficient,
/// and only reallocated when growing or shrinking significantly.
pub struct EdgeRenderer {
    vertex_buf: Option<wgpu::Buffer>,
    index_buf: Option<wgpu::Buffer>,
    vertex_capacity: u64,
    index_capacity: u64,
    count: u32, // number of indices
}

impl EdgeRenderer {
    pub fn new() -> Self {
        Self {
            vertex_buf: None,
            index_buf: None,
            vertex_capacity: 0,
            index_capacity: 0,
            count: 0,
        }
    }

    /// Re-upload edge data to the GPU, reusing buffers when possible.
    pub fn update(&mut self, edges: &[GpuEdge], device: &wgpu::Device, queue: &wgpu::Queue) {
        if edges.is_empty() {
            self.count = 0;
            return;
        }

        // Expand each GpuEdge into 4 vertices.
        // Side encoding: sign = which endpoint, magnitude = which side of the line.
        // +2.0 = start, left | +1.0 = start, right | -1.0 = end, right | -2.0 = end, left
        // Must match thresholds in edge.wgsl (1.5, 0.5, -0.5).
        const SIDE_START_LEFT: f32 = 2.0;
        const SIDE_START_RIGHT: f32 = 1.0;
        const SIDE_END_RIGHT: f32 = -1.0;
        const SIDE_END_LEFT: f32 = -2.0;
        let side_values = [
            SIDE_START_LEFT,
            SIDE_START_RIGHT,
            SIDE_END_RIGHT,
            SIDE_END_LEFT,
        ];
        let mut verts: Vec<EdgeVertex> = Vec::with_capacity(edges.len() * 4);
        let mut indices: Vec<u32> = Vec::with_capacity(edges.len() * 6);

        for (ei, edge) in edges.iter().enumerate() {
            let base = (ei * 4) as u32;
            for &side in &side_values {
                verts.push(EdgeVertex {
                    from_pos: edge.from_pos,
                    to_pos: edge.to_pos,
                    color: edge.color,
                    side,
                });
            }
            // two triangles: (0,1,2) and (0,2,3)
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        self.count = indices.len() as u32;

        let vert_bytes = bytemuck::cast_slice::<EdgeVertex, u8>(&verts);
        let idx_bytes = bytemuck::cast_slice::<u32, u8>(&indices);
        let vert_need = vert_bytes.len() as u64;
        let idx_need = idx_bytes.len() as u64;

        let should_realloc_verts = self.vertex_capacity < vert_need
            || (self.vertex_capacity > vert_need.saturating_mul(4)
                && self.vertex_capacity > 64 * 1024);
        let should_realloc_idx = self.index_capacity < idx_need
            || (self.index_capacity > idx_need.saturating_mul(4)
                && self.index_capacity > 64 * 1024);

        if should_realloc_verts {
            self.vertex_buf = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("edge_vertices"),
                    contents: vert_bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
            );
            self.vertex_capacity = vert_need;
        } else {
            queue.write_buffer(
                self.vertex_buf
                    .as_ref()
                    .expect("vertex buffer must exist when capacity > 0"),
                0,
                vert_bytes,
            );
        }

        if should_realloc_idx {
            self.index_buf = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("edge_indices"),
                    contents: idx_bytes,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                }),
            );
            self.index_capacity = idx_need;
        } else {
            queue.write_buffer(
                self.index_buf
                    .as_ref()
                    .expect("index buffer must exist when capacity > 0"),
                0,
                idx_bytes,
            );
        }
    }

    /// Draw all edges. Caller must have already set the edge pipeline + bind group.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.count == 0 {
            return;
        }
        let Some(vb) = &self.vertex_buf else { return };
        let Some(ib) = &self.index_buf else { return };
        pass.set_vertex_buffer(0, vb.slice(..));
        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.count, 0, 0..1);
    }
}

impl Default for EdgeRenderer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_edge_size_matches_layout() {
        // GpuEdge: 3 floats (from) + 3 floats (to) + 4 floats (color) = 10 * 4 = 40 bytes
        assert_eq!(
            std::mem::size_of::<GpuEdge>(),
            40,
            "GpuEdge should be 40 bytes"
        );
    }

    #[test]
    fn edge_vertex_size_matches_stride() {
        // EdgeVertex: 3+3+4+1 = 11 floats = 44 bytes, matching EDGE_VERTEX_LAYOUT stride
        assert_eq!(
            std::mem::size_of::<EdgeVertex>(),
            44,
            "EdgeVertex should be 44 bytes (matches shader stride)"
        );
        assert_eq!(EDGE_VERTEX_LAYOUT.array_stride, 44);
    }

    #[test]
    fn edge_vertex_attributes_cover_all_fields() {
        // 4 attributes: from_pos(0), to_pos(12), color(24), side(40)
        assert_eq!(EDGE_VERTEX_ATTRS.len(), 4);
        assert_eq!(EDGE_VERTEX_ATTRS[0].offset, 0);
        assert_eq!(EDGE_VERTEX_ATTRS[1].offset, 12);
        assert_eq!(EDGE_VERTEX_ATTRS[2].offset, 24);
        assert_eq!(EDGE_VERTEX_ATTRS[3].offset, 40);
    }

    #[test]
    fn new_edge_renderer_has_zero_count() {
        let er = EdgeRenderer::new();
        assert_eq!(er.count, 0);
        assert!(er.vertex_buf.is_none());
        assert!(er.index_buf.is_none());
    }
}
