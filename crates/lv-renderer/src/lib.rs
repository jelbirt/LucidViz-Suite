//! `lv-renderer` — GPU rendering layer for Lucid Visualization Suite.
//!
//! Public API:
//!  - `WgpuContext`     — wgpu device/surface wrapper
//!  - `ArcballCamera`   — orbit camera
//!  - `FrameTimer`      — LIS slice advance timer
//!  - `InstanceBuffer`  — per-shape GPU buffers
//!  - `build_lis_buffer` / `compute_frame` — LIS animation builder
//!  - `render_headless` — off-screen render for testing
//!  - `shapes::*`       — mesh generators

pub mod camera;
pub mod device;
pub mod edge_renderer;
pub mod frame_timer;
pub mod instanced;
pub mod lis;
pub mod pipelines;
pub mod shapes;

pub use camera::{AppAction, ArcballCamera, CameraKey};
pub use device::WgpuContext;
pub use edge_renderer::{EdgeRenderer, GpuEdge, EDGE_VERTEX_LAYOUT};
pub use frame_timer::{FrameAdvance, FrameTimer};
pub use instanced::InstanceBuffer;
pub use lis::{build_lis_buffer, compute_frame};
pub use shapes::{GpuMesh, Lod, Vertex};

// ─── Uniforms (shared across pipelines) ──────────────────────────────────────

/// GPU uniform block — must match the WGSL `Uniforms` structs in the shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShapeUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub ambient: f32,
}

/// GPU uniform block for the edge shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EdgeUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub viewport_size: [f32; 2],
    pub _pad: [f32; 2],
}

// ─── Headless rendering (used in tests) ──────────────────────────────────────

/// Render `frame` off-screen and return the raw RGBA8 pixel bytes.
///
/// Uses a wgpu instance with no surface (headless). Suitable for golden-hash
/// tests. Blocks the calling thread (uses `pollster`).
#[cfg(any(test, feature = "headless"))]
pub fn render_headless(frame: &lv_data::LisFrame, width: u32, height: u32) -> Vec<u8> {
    pollster::block_on(render_headless_async(frame, width, height))
}

#[cfg(any(test, feature = "headless"))]
async fn render_headless_async(frame: &lv_data::LisFrame, width: u32, height: u32) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("no headless adapter");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("headless"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .expect("headless device");

    // Render target texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_rt"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Depth texture
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("headless_shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../assets/shaders/shape_instanced.wgsl").into(),
        ),
    });

    // Uniform buffer: identity view_proj
    let camera = ArcballCamera::new(width as f32 / height as f32);
    let vp: [[f32; 4]; 4] = camera.view_proj().into();
    let uniforms = ShapeUniforms {
        view_proj: vp,
        light_dir: [0.577, 0.577, 0.577],
        ambient: 0.15,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("headless_uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bgl = pipelines::uniform_bind_group_layout(&device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("headless_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    // Build meshes (LOD Low) and upload instances
    let mut instance_buf = InstanceBuffer::new();
    use lv_data::ShapeKind;
    let mut shape_instances: std::collections::HashMap<u32, Vec<lv_data::GpuInstance>> =
        std::collections::HashMap::new();
    for inst in &frame.instances {
        shape_instances
            .entry(inst.shape_id)
            .or_default()
            .push(*inst);
    }

    for sk in ShapeKind::ALL {
        let insts = shape_instances
            .get(&sk.gpu_id())
            .cloned()
            .unwrap_or_default();
        if !insts.is_empty() {
            instance_buf.update(sk, &insts, &device, &queue);
        }
    }

    // Fake SurfaceConfiguration for pipeline creation (headless needs format)
    let fake_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let pipeline = {
        // Build a minimal WgpuContext-like thing just to reuse create_shape_pipeline.
        // Instead we build the pipeline directly here.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hl_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hl_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[shapes::Vertex::LAYOUT, pipelines::instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: fake_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
    };

    // Build meshes
    let meshes: Vec<GpuMesh> = ShapeKind::ALL
        .iter()
        .map(|sk| {
            let (verts, idx) = match sk {
                ShapeKind::Sphere => shapes::sphere::build(Lod::Low),
                ShapeKind::Point => shapes::point::build(Lod::Low),
                ShapeKind::Torus => shapes::torus::build(Lod::Low),
                ShapeKind::Pyramid => shapes::pyramid::build(Lod::Low),
                ShapeKind::Cube => shapes::cube::build(Lod::Low),
                ShapeKind::Cylinder => shapes::cylinder::build(Lod::Low),
            };
            GpuMesh::upload(&device, &format!("headless_{sk:?}"), &verts, &idx)
        })
        .collect();

    // Render pass
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("headless_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.05,
                        b: 0.08,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        for (i, sk) in ShapeKind::ALL.iter().enumerate() {
            if let Some(insts) = shape_instances.get(&sk.gpu_id()) {
                instance_buf.draw(*sk, &meshes[i], &mut pass, insts.len() as u32);
            }
        }
    }

    // Copy texture to buffer for readback
    let bytes_per_row = 4 * width;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bpr = bytes_per_row.div_ceil(align) * align;
    let buf_size = (padded_bpr * height) as u64;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless_readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Map and collect
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("device poll failed");
    rx.recv().unwrap().expect("readback map failed");

    let raw = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        pixels.extend_from_slice(&raw[start..start + (bytes_per_row as usize)]);
    }
    drop(raw);
    readback.unmap();
    pixels
}
