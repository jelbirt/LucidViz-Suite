//! Per-shape GPU instance buffers.
//!
//! One `wgpu::Buffer` is maintained per `ShapeKind`. Buffers grow as needed and
//! can compact after large downshifts in instance count.

use lv_data::{GpuInstance, ShapeKind};

const SHAPE_COUNT: usize = ShapeKind::ALL.len(); // 6

/// Holds one resizable instance buffer per `ShapeKind`.
pub struct InstanceBuffer {
    buffers: [Option<wgpu::Buffer>; SHAPE_COUNT],
    capacities: [u64; SHAPE_COUNT],
    counts: [u32; SHAPE_COUNT],
}

impl InstanceBuffer {
    pub fn new() -> Self {
        Self {
            buffers: [const { None }; SHAPE_COUNT],
            capacities: [0; SHAPE_COUNT],
            counts: [0; SHAPE_COUNT],
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
        let idx = shape as usize;
        if instances.is_empty() {
            self.counts[idx] = 0;
            return;
        }

        let bytes = bytemuck::cast_slice::<GpuInstance, u8>(instances);
        let need = bytes.len() as u64;
        self.counts[idx] = instances.len() as u32;

        let should_shrink = self.capacities[idx] > need.saturating_mul(4)
            && self.capacities[idx] > (64 * 1024) as u64;

        if self.capacities[idx] < need || should_shrink {
            // Allocate with 2× headroom (min 64KB) to amortize repeated
            // small growths when node count changes across frames.
            let alloc = need.next_power_of_two().max(64 * 1024);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("inst_buf_{idx}")),
                size: alloc,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytes);
            self.capacities[idx] = alloc;
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
    pub fn instance_count(&self, shape: ShapeKind) -> u32 {
        self.counts[shape as usize]
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
