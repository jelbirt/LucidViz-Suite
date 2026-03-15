//! Render pipeline factory functions.

use crate::device::WgpuContext;
use crate::shapes::Vertex;
use lv_data::GpuInstance;

/// Vertex buffer layout for per-instance `GpuInstance` data (64 bytes).
///
/// Shader locations 2–8 map to the GpuInstance fields.
pub fn instance_layout() -> wgpu::VertexBufferLayout<'static> {
    use std::mem::size_of;
    use wgpu::{VertexAttribute, VertexFormat::*, VertexStepMode};

    // The attributes must match the `InstanceInput` struct in shape_instanced.wgsl.
    static ATTRS: &[VertexAttribute] = &[
        // location 2: inst_position   vec3<f32>   @ offset 0
        VertexAttribute {
            shader_location: 2,
            offset: 0,
            format: Float32x3,
        },
        // location 3: inst_size       f32         @ offset 12
        VertexAttribute {
            shader_location: 3,
            offset: 12,
            format: Float32,
        },
        // location 4: inst_size_alpha f32         @ offset 16
        VertexAttribute {
            shader_location: 4,
            offset: 16,
            format: Float32,
        },
        // location 5: inst_pad0       vec3<f32>   @ offset 20  (padding)
        VertexAttribute {
            shader_location: 5,
            offset: 20,
            format: Float32x3,
        },
        // location 6: inst_color      vec4<f32>   @ offset 32
        VertexAttribute {
            shader_location: 6,
            offset: 32,
            format: Float32x4,
        },
        // location 7: inst_spin       vec3<f32>   @ offset 48
        VertexAttribute {
            shader_location: 7,
            offset: 48,
            format: Float32x3,
        },
        // location 8: inst_shape_id   u32         @ offset 60
        VertexAttribute {
            shader_location: 8,
            offset: 60,
            format: Uint32,
        },
    ];

    wgpu::VertexBufferLayout {
        array_stride: size_of::<GpuInstance>() as wgpu::BufferAddress,
        step_mode: VertexStepMode::Instance,
        attributes: ATTRS,
    }
}

/// Uniform bind-group layout shared by both pipelines.
pub fn uniform_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("uniform_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// Opaque shape pipeline (depth write ON).
pub fn create_shape_pipeline(
    ctx: &WgpuContext,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape_pipeline_layout"),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shape_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT, instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: WgpuContext::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
}

/// Opaque shape pipeline using explicit format — for headless/offscreen rendering.
pub fn create_shape_pipeline_with_format(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shape_pl_layout_fmt"),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("shape_pipeline_fmt"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[Vertex::LAYOUT, instance_layout()],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
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
            format: depth_format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Axis / line pipeline — `LineList` topology, unlit, using explicit formats.
///
/// Vertex layout (24 bytes per vertex):
///   location 0: position  vec3<f32>  @ offset 0
///   location 1: color     vec4<f32>  @ offset 12
pub fn create_axis_pipeline_with_format(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    static AXIS_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x4,
    ];
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("axis_pl_layout"),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("axis_pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: (3 + 4) * 4, // 28 bytes
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: AXIS_ATTRS,
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            // Write depth so axis lines occlude/are occluded correctly.
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Edge pipeline: alpha blending, depth write OFF.
pub fn create_edge_pipeline(
    ctx: &WgpuContext,
    shader: &wgpu::ShaderModule,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("edge_pipeline_layout"),
            bind_group_layouts: &[bgl],
            push_constant_ranges: &[],
        });

    // Per-edge-vertex layout (8 floats = 32 bytes):
    //   location 0: from_pos  vec3
    //   location 1: to_pos    vec3
    //   location 2: color     vec4
    //   location 3: side      f32
    static EDGE_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x4,
        3 => Float32
    ];

    let edge_vertex_layout = wgpu::VertexBufferLayout {
        array_stride: (3 + 3 + 4 + 1) * 4, // 44 bytes
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: EDGE_ATTRS,
    };

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                buffers: &[edge_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ctx.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: WgpuContext::DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
}
