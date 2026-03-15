//! Per-shape GPU instance buffers.
//!
//! One `wgpu::Buffer` is maintained per `ShapeKind`. Buffers are grown (but
//! never shrunk) as needed when new instance data arrives.

use lv_data::{GpuInstance, ShapeKind};
use wgpu::util::DeviceExt;

const SHAPE_COUNT: usize = ShapeKind::ALL.len(); // 6

/// Holds one resizable instance buffer per `ShapeKind`.
pub struct InstanceBuffer {
    buffers: [Option<wgpu::Buffer>; SHAPE_COUNT],
    capacities: [u64; SHAPE_COUNT],
}

impl InstanceBuffer {
    pub fn new() -> Self {
        Self {
            buffers: [const { None }; SHAPE_COUNT],
            capacities: [0; SHAPE_COUNT],
        }
    }

    /// Write `instances` for `shape` to the GPU.  Reallocates if needed.
    pub fn update(
        &mut self,
        shape: ShapeKind,
        instances: &[GpuInstance],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        if instances.is_empty() {
            return;
        }

        let idx = shape as usize;
        let bytes = bytemuck::cast_slice::<GpuInstance, u8>(instances);
        let need = bytes.len() as u64;

        if self.capacities[idx] < need {
            // (Re)allocate a buffer large enough
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("inst_buf_{idx}")),
                contents: bytes,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.capacities[idx] = need;
            self.buffers[idx] = Some(buf);
        } else {
            queue.write_buffer(
                self.buffers[idx]
                    .as_ref()
                    .expect("buffer must exist: allocated above"),
                0,
                bytes,
            );
        }
    }

    /// Number of instances that can be drawn for `shape` (based on what was last updated).
    pub fn instance_count(&self, shape: ShapeKind, instances: &[GpuInstance]) -> u32 {
        let _ = shape;
        instances.len() as u32
    }

    /// Set the instance buffer for `shape` and record a `draw_indexed` call.
    pub fn draw<'a>(
        &'a self,
        shape: ShapeKind,
        mesh: &'a crate::shapes::GpuMesh,
        pass: &mut wgpu::RenderPass<'a>,
        inst_count: u32,
    ) {
        if inst_count == 0 {
            return;
        }
        let idx = shape as usize;
        if let Some(buf) = &self.buffers[idx] {
            pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, buf.slice(..));
            pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..inst_count);
        }
    }
}

impl Default for InstanceBuffer {
    fn default() -> Self {
        Self::new()
    }
}
