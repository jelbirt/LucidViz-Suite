//! `lv-app` — Lucid Visualization Suite entry-point.
//!
//! Integrates wgpu rendering (Phase 4) with egui immediate-mode GUI (Phase 5)
//! and the Ego Cluster system (Phase 7).

mod app_state;
mod notifications;
mod prefs;
mod session;

use std::sync::Arc;

use app_state::{build_gpu_edges, compute_ego_edges, compute_visible_objects, EgoClusterState};
use lv_data::{EtvDataset, EtvRow, EtvSheet, GpuInstance, LisBuffer, LisConfig, ShapeKind};
use lv_gui::{AppState, LucidWorkspace};
use lv_renderer::shapes::{self, cube, cylinder, point, pyramid, sphere, torus};
use lv_renderer::{
    build_lis_buffer, compute_frame, AppAction, ArcballCamera, CameraKey, EdgeRenderer,
    EdgeUniforms, FrameTimer, GpuMesh, InstanceBuffer, Lod, ShapeUniforms, WgpuContext,
    EDGE_VERTEX_LAYOUT,
};
use notifications::NotificationQueue;
use prefs::UserPreferences;

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// ── GpuInstance vertex buffer layout ─────────────────────────────────────────
static INSTANCE_ATTRS: &[wgpu::VertexAttribute] = &[
    wgpu::VertexAttribute {
        shader_location: 2,
        offset: 0,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        shader_location: 3,
        offset: 12,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        shader_location: 4,
        offset: 16,
        format: wgpu::VertexFormat::Float32,
    },
    wgpu::VertexAttribute {
        shader_location: 5,
        offset: 20,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        shader_location: 6,
        offset: 32,
        format: wgpu::VertexFormat::Float32x4,
    },
    wgpu::VertexAttribute {
        shader_location: 7,
        offset: 48,
        format: wgpu::VertexFormat::Float32x3,
    },
    wgpu::VertexAttribute {
        shader_location: 8,
        offset: 60,
        format: wgpu::VertexFormat::Uint32,
    },
];

const INSTANCE_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: 64,
    step_mode: wgpu::VertexStepMode::Instance,
    attributes: INSTANCE_ATTRS,
};

// ── demo dataset ─────────────────────────────────────────────────────────────
//
// Global Trade Network, 2017-2024
// ─────────────────────────────────────────────────────────────────────────────
// 48 country-nodes in 5 regional clusters.  Each of the 8 sheets represents
// one calendar year.  Node positions drift over time to show structural change
// (e.g. US-China decoupling, post-Brexit UK drift, emerging-market rise).
//
// Encoding:
//   Shape   → region  (Sphere=Americas, Cube=Europe, Cylinder=Asia-Pacific,
//                       Torus=Africa/MENA, Pyramid=Central/South Asia)
//   Color   → region palette (saturated, distinct)
//   Size    → GDP tier (0.18 – 0.55)
//   Cluster → GDP tier 0-4
//   Spin    → trade velocity (large economies spin faster on Y-axis)
//   Edges   → top bilateral trade relationships, strength ∝ trade volume

fn make_demo_dataset() -> EtvDataset {
    // ── node catalogue ────────────────────────────────────────────────────────
    // (label, region, gdp_tier 0-4, base_x, base_y, base_z, spin_y deg/frame)
    #[allow(clippy::type_complexity)]
    let nodes: &[(&str, usize, u32, f64, f64, f64, f64)] = &[
        // Americas  (region 0)  shape=Sphere  colour=blue
        ("USA", 0, 4, 4.0, 1.5, 0.5, 0.8),
        ("China", 2, 4, -4.0, 1.0, 0.0, 0.8), // placed Asia but biggest trade partner
        ("Canada", 0, 2, 3.5, 3.0, 1.0, 0.3),
        ("Mexico", 0, 2, 3.0, -0.5, -0.5, 0.3),
        ("Brazil", 0, 3, 2.0, -2.5, 1.5, 0.5),
        ("Argentina", 0, 1, 1.5, -3.5, 0.5, 0.2),
        ("Colombia", 0, 1, 2.5, -2.0, -1.0, 0.2),
        ("Chile", 0, 1, 1.0, -3.0, 2.0, 0.2),
        ("Peru", 0, 0, 1.5, -2.8, 2.5, 0.1),
        ("Venezuela", 0, 0, 2.0, -1.5, -2.0, 0.1),
        // Europe  (region 1)  shape=Cube  colour=green
        ("Germany", 1, 3, -1.0, 3.5, -1.5, 0.6),
        ("France", 1, 3, -0.5, 3.0, -2.5, 0.5),
        ("UK", 1, 3, 0.5, 3.5, -3.0, 0.5),
        ("Italy", 1, 2, 0.0, 2.5, -2.0, 0.4),
        ("Spain", 1, 2, -0.5, 2.0, -3.0, 0.3),
        ("Netherlands", 1, 2, -1.5, 3.8, -0.5, 0.4),
        ("Belgium", 1, 1, -1.8, 3.2, -1.0, 0.2),
        ("Sweden", 1, 1, -0.8, 4.5, -1.0, 0.2),
        ("Poland", 1, 1, 0.2, 4.0, -0.5, 0.2),
        ("Switzerland", 1, 2, -0.3, 3.0, -1.5, 0.3),
        // Asia-Pacific  (region 2)  shape=Cylinder  colour=red
        ("Japan", 2, 3, -3.0, 2.0, 2.0, 0.6),
        ("South Korea", 2, 2, -3.5, 1.5, 3.0, 0.5),
        ("India", 2, 3, -2.0, 0.0, 3.5, 0.6),
        ("Australia", 2, 2, -2.5, -2.0, 4.0, 0.3),
        ("Indonesia", 2, 2, -3.5, -1.0, 4.5, 0.3),
        ("Thailand", 2, 1, -4.0, -0.5, 3.5, 0.2),
        ("Vietnam", 2, 1, -4.5, 0.0, 3.0, 0.2),
        ("Malaysia", 2, 1, -4.0, -1.5, 3.0, 0.2),
        ("Philippines", 2, 1, -3.8, -2.0, 2.5, 0.2),
        ("New Zealand", 2, 0, -2.0, -3.0, 4.5, 0.1),
        // Africa / MENA  (region 3)  shape=Torus  colour=orange
        ("Saudi Arabia", 3, 3, 0.5, 0.0, -4.0, 0.5),
        ("UAE", 3, 2, 1.0, 0.5, -4.5, 0.4),
        ("Turkey", 3, 2, 0.5, 1.5, -3.5, 0.3),
        ("South Africa", 3, 1, -1.0, -1.5, -3.5, 0.2),
        ("Nigeria", 3, 1, -0.5, -2.0, -4.0, 0.2),
        ("Egypt", 3, 1, 0.0, -0.5, -4.5, 0.2),
        ("Morocco", 3, 0, -0.5, 0.0, -5.0, 0.1),
        ("Kenya", 3, 0, -1.5, -2.5, -3.0, 0.1),
        ("Ethiopia", 3, 0, -1.0, -3.0, -3.5, 0.1),
        ("Ghana", 3, 0, 0.0, -2.8, -4.5, 0.1),
        // Central / South Asia  (region 4)  shape=Pyramid  colour=purple
        ("Russia", 4, 3, -2.0, 2.5, -1.0, 0.5),
        ("Kazakhstan", 4, 1, -2.5, 1.5, -2.0, 0.2),
        ("Pakistan", 4, 1, -1.5, 0.5, 2.5, 0.2),
        ("Bangladesh", 4, 0, -1.0, 0.0, 4.0, 0.1),
        ("Sri Lanka", 4, 0, -1.5, -0.5, 4.5, 0.1),
        ("Uzbekistan", 4, 0, -2.0, 1.0, -1.5, 0.1),
        ("Iran", 4, 1, -0.5, 0.5, -3.0, 0.2),
        ("Iraq", 4, 0, 0.0, 0.0, -3.5, 0.1),
    ];

    // ── region visual encoding ────────────────────────────────────────────────
    let region_shape = [
        ShapeKind::Sphere,   // Americas
        ShapeKind::Cube,     // Europe
        ShapeKind::Cylinder, // Asia-Pacific
        ShapeKind::Torus,    // Africa/MENA
        ShapeKind::Pyramid,  // Central/South Asia
    ];
    // RGBA palette per region
    let region_color: [(f32, f32, f32); 5] = [
        (0.25, 0.55, 0.95), // blue   – Americas
        (0.25, 0.80, 0.45), // green  – Europe
        (0.95, 0.30, 0.30), // red    – Asia-Pacific
        (0.95, 0.65, 0.15), // orange – Africa/MENA
        (0.75, 0.35, 0.95), // purple – Central/South Asia
    ];
    // GDP tier → display size
    let tier_size = [0.18_f64, 0.25, 0.33, 0.44, 0.55];

    // ── top bilateral trade edges (fixed topology, strengths grow over time) ──
    // (from, to, base_strength, growth_per_year)
    let edge_defs: &[(&str, &str, f64, f64)] = &[
        ("USA", "Canada", 0.90, 0.01),
        ("USA", "Mexico", 0.85, 0.01),
        ("USA", "China", 0.80, -0.04), // US-China decoupling
        ("USA", "Japan", 0.70, 0.00),
        ("USA", "Germany", 0.65, 0.00),
        ("USA", "UK", 0.60, 0.01),
        ("China", "Japan", 0.75, -0.02),
        ("China", "South Korea", 0.78, -0.01),
        ("China", "Germany", 0.70, 0.00),
        ("China", "Australia", 0.65, -0.03),
        ("China", "Vietnam", 0.55, 0.05), // supply-chain shift
        ("China", "India", 0.50, -0.03),
        ("Germany", "France", 0.88, 0.00),
        ("Germany", "Netherlands", 0.80, 0.00),
        ("Germany", "Italy", 0.75, 0.00),
        ("Germany", "UK", 0.72, -0.02),     // post-Brexit drift
        ("Germany", "Poland", 0.60, 0.03),  // nearshoring
        ("Germany", "Russia", 0.55, -0.10), // sanctions
        ("UK", "France", 0.68, -0.02),
        ("UK", "Netherlands", 0.65, -0.01),
        ("Japan", "South Korea", 0.72, 0.00),
        ("Japan", "Australia", 0.65, 0.01),
        ("Japan", "India", 0.50, 0.03),
        ("India", "UAE", 0.60, 0.04),
        ("India", "Saudi Arabia", 0.55, 0.02),
        ("India", "USA", 0.58, 0.03),
        ("Brazil", "China", 0.65, 0.04),
        ("Brazil", "USA", 0.55, 0.00),
        ("Russia", "China", 0.70, 0.08), // Russia pivot East
        ("Russia", "India", 0.45, 0.06),
        ("Saudi Arabia", "China", 0.68, 0.05),
        ("Saudi Arabia", "India", 0.62, 0.03),
        ("UAE", "India", 0.58, 0.04),
        ("Vietnam", "USA", 0.52, 0.06),
        ("Vietnam", "South Korea", 0.55, 0.04),
        ("South Korea", "USA", 0.65, 0.00),
        ("Australia", "USA", 0.58, 0.01),
        ("Indonesia", "China", 0.55, 0.03),
        ("Mexico", "Canada", 0.65, 0.02),
        ("Turkey", "Germany", 0.60, 0.01),
    ];

    // ── time-series drift vectors (per-year displacement, in 3-D) ────────────
    // Applied cumulatively: position at year t = base + t * drift
    // Drift encodes real-world structural shifts:
    //   • UK drifts away from Europe cluster (Brexit)
    //   • Russia drifts toward China/Asia (sanctions pivot)
    //   • Vietnam/Indonesia drift toward centre (supply-chain rise)
    let drift: std::collections::HashMap<&str, (f64, f64, f64)> = [
        ("UK", (-0.15, -0.05, 0.08)),     // away from EU
        ("Russia", (-0.35, 0.00, 0.30)),  // toward Asia
        ("Vietnam", (0.25, 0.05, -0.15)), // toward centre
        ("Indonesia", (0.20, 0.03, -0.10)),
        ("India", (-0.10, 0.05, 0.05)), // rising centrality
        ("Germany", (0.00, 0.00, 0.05)),
        ("China", (0.10, 0.05, -0.05)), // slight centralisation
        ("USA", (-0.05, -0.02, 0.02)),
        ("Saudi Arabia", (0.05, 0.05, 0.10)), // more connected
        ("UAE", (0.05, 0.05, 0.10)),
        ("Brazil", (0.05, 0.03, 0.00)),
        ("Poland", (-0.10, 0.00, -0.05)),    // toward EU core
        ("Kazakhstan", (-0.10, 0.00, 0.15)), // follows Russia
    ]
    .iter()
    .cloned()
    .collect();

    let num_sheets: usize = 8;
    let all_labels: Vec<String> = nodes.iter().map(|(l, ..)| l.to_string()).collect();

    let sheets: Vec<EtvSheet> = (0..num_sheets)
        .map(|s| {
            let year_offset = s as f64;

            let rows: Vec<EtvRow> = nodes
                .iter()
                .map(|(label, region, gdp_tier, bx, by, bz, spin_y)| {
                    let (dx, dy, dz) = drift.get(label).copied().unwrap_or((0.0, 0.0, 0.0));

                    // Add small per-year noise so non-drifting nodes still
                    // animate slightly (±0.08 based on deterministic hash)
                    let h = label
                        .bytes()
                        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64));
                    let noise_scale = 0.08;
                    let nx = noise_scale
                        * ((h.wrapping_mul(1_000_003).wrapping_add(s as u64 * 7)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;
                    let ny = noise_scale
                        * ((h.wrapping_mul(1_000_033).wrapping_add(s as u64 * 13)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;
                    let nz = noise_scale
                        * ((h.wrapping_mul(1_000_099).wrapping_add(s as u64 * 19)) as f64
                            / u64::MAX as f64
                            - 0.5)
                        * 2.0;

                    let (cr, cg, cb) = region_color[*region];
                    EtvRow {
                        label: label.to_string(),
                        x: bx + dx * year_offset + nx,
                        y: by + dy * year_offset + ny,
                        z: bz + dz * year_offset + nz,
                        size: tier_size[*gdp_tier as usize],
                        shape: region_shape[*region],
                        color_r: cr,
                        color_g: cg,
                        color_b: cb,
                        cluster_value: *gdp_tier as f64,
                        spin_y: *spin_y,
                        ..EtvRow::default()
                    }
                })
                .collect();

            // Compute edges for this time slice with strength drift
            let edges: Vec<lv_data::EdgeRow> = edge_defs
                .iter()
                .filter_map(|(from, to, base, growth)| {
                    let strength = (base + growth * year_offset).clamp(0.0, 1.0);
                    if strength > 0.1 {
                        Some(lv_data::EdgeRow {
                            from: from.to_string(),
                            to: to.to_string(),
                            strength,
                        })
                    } else {
                        None
                    }
                })
                .collect();

            EtvSheet {
                name: format!("{}", 2017 + s),
                sheet_index: s,
                rows,
                edges,
            }
        })
        .collect();

    EtvDataset {
        source_path: None,
        sheets,
        all_labels,
    }
}

// ── per-shape mesh table ──────────────────────────────────────────────────────

struct ShapeMeshes {
    kinds: Vec<ShapeKind>,
    meshes: Vec<GpuMesh>,
}

impl ShapeMeshes {
    fn build(device: &wgpu::Device, lod: Lod) -> Self {
        #[allow(clippy::type_complexity)]
        let raw: Vec<(ShapeKind, &str, (Vec<shapes::Vertex>, Vec<u32>))> = vec![
            (ShapeKind::Sphere, "sphere", sphere::build(lod)),
            (ShapeKind::Cube, "cube", cube::build(lod)),
            (ShapeKind::Cylinder, "cylinder", cylinder::build(lod)),
            (ShapeKind::Torus, "torus", torus::build(lod)),
            (ShapeKind::Pyramid, "pyramid", pyramid::build(lod)),
            (ShapeKind::Point, "point", point::build(lod)),
        ];
        let mut kinds = Vec::with_capacity(raw.len());
        let mut meshes = Vec::with_capacity(raw.len());
        for (kind, label, (verts, idxs)) in raw {
            kinds.push(kind);
            meshes.push(GpuMesh::upload(device, label, &verts, &idxs));
        }
        Self { kinds, meshes }
    }
}

// ── Renderer ─────────────────────────────────────────────────────────────────

struct Renderer {
    ctx: WgpuContext,
    camera: ArcballCamera,
    timer: FrameTimer,
    inst_buf: InstanceBuffer,
    shape_meshes: ShapeMeshes,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    lis_buffer: LisBuffer,
    dataset: EtvDataset,
    lis_config: LisConfig,
    slice_index: u32,
    workspace: LucidWorkspace,
    app_state: AppState,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    // Phase 7: Ego Cluster system
    ego_state: EgoClusterState,
    edge_renderer: EdgeRenderer,
    edge_pipeline: wgpu::RenderPipeline,
    edge_bind_group: wgpu::BindGroup,
    edge_uniform_buf: wgpu::Buffer,
    window_size: (u32, u32),
    last_click_pos: Option<(f64, f64)>,
    // Phase 9: preferences + notifications
    prefs: UserPreferences,
    notifications: NotificationQueue,
}

impl Renderer {
    fn new(window: &Window) -> Self {
        let ctx = WgpuContext::new(window).expect("WgpuContext");

        let dataset = make_demo_dataset();
        let lis_config = LisConfig::default();
        let lis_buffer = build_lis_buffer(&dataset, &lis_config);

        let size = window.inner_size();
        let aspect = size.width as f32 / size.height.max(1) as f32;
        let camera = ArcballCamera::new(aspect);
        let timer = FrameTimer::new();

        // ── Shape uniform buffer + bind group ─────────────────────────────────
        let uniforms = ShapeUniforms {
            view_proj: camera.view_proj().into(),
            light_dir: [0.577, 0.577, 0.577],
            ambient: 0.15,
        };
        let uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shape_uniforms"),
                contents: bytemuck::bytes_of(&uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shape_bgl"),
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
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shape_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // ── Shape pipeline ────────────────────────────────────────────────────
        let shader_src = include_str!("../../../assets/shaders/shape_instanced.wgsl");
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shape_instanced"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
        let pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shape_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shape_pipeline"),
                layout: Some(&pl_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[shapes::Vertex::LAYOUT, INSTANCE_LAYOUT],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // ── Edge uniform buffer + bind group ──────────────────────────────────
        let edge_uniforms = EdgeUniforms {
            view_proj: camera.view_proj().into(),
            viewport_size: [size.width as f32, size.height as f32],
            _pad: [0.0; 2],
        };
        let edge_uniform_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge_uniforms"),
                contents: bytemuck::bytes_of(&edge_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let edge_bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("edge_bgl"),
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
            });
        let edge_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("edge_bg"),
            layout: &edge_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: edge_uniform_buf.as_entire_binding(),
            }],
        });

        // ── Edge pipeline ─────────────────────────────────────────────────────
        let edge_shader_src = include_str!("../../../assets/shaders/edge.wgsl");
        let edge_shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("edge"),
                source: wgpu::ShaderSource::Wgsl(edge_shader_src.into()),
            });
        let edge_pl_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("edge_pl"),
                bind_group_layouts: &[&edge_bgl],
                push_constant_ranges: &[],
            });
        let edge_pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("edge_pipeline"),
                layout: Some(&edge_pl_layout),
                vertex: wgpu::VertexState {
                    module: &edge_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[EDGE_VERTEX_LAYOUT],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &edge_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ctx.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        let depth_view = Self::make_depth_view(&ctx.device, size.width.max(1), size.height.max(1));
        let shape_meshes = ShapeMeshes::build(&ctx.device, Lod::Mid);
        let inst_buf = InstanceBuffer::new();

        // egui
        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &ctx.device,
            ctx.config.format,
            egui_wgpu::RendererOptions {
                // Egui draws 2D UI; it does not need a depth attachment. Using `None`
                // keeps the egui render pipeline compatible with the main 3D render
                // pass which provides a depth texture separately.
                depth_stencil_format: None,
                msaa_samples: 1,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );

        Self {
            ctx,
            camera,
            timer,
            inst_buf,
            shape_meshes,
            pipeline,
            bind_group,
            uniform_buf,
            depth_view,
            lis_buffer,
            dataset,
            lis_config,
            slice_index: 0,
            workspace: LucidWorkspace::new(),
            app_state: AppState::new(),
            egui_ctx,
            egui_winit,
            egui_renderer,
            ego_state: EgoClusterState::default(),
            edge_renderer: EdgeRenderer::new(),
            edge_pipeline,
            edge_bind_group,
            edge_uniform_buf,
            window_size: (size.width, size.height),
            last_click_pos: None,
            prefs: UserPreferences::load(),
            notifications: NotificationQueue::default(),
        }
    }

    fn make_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.ctx.resize(winit::dpi::PhysicalSize::new(w, h));
        self.depth_view = Self::make_depth_view(&self.ctx.device, w, h);
        self.camera.set_aspect(w, h);
        self.window_size = (w, h);
    }

    fn rebuild_lis(&mut self) {
        self.lis_buffer = build_lis_buffer(&self.dataset, &self.lis_config);
        self.slice_index = 0;
    }

    /// Object picking: project world-space positions to screen, find closest to
    /// `click` within 20 px.  `label_positions` is a `(label, world_pos)` list.
    fn pick_object(
        &self,
        click: (f64, f64),
        label_positions: &[(String, [f32; 3])],
    ) -> Option<String> {
        let (w, h) = self.window_size;
        if w == 0 || h == 0 {
            return None;
        }

        let view_proj: nalgebra::Matrix4<f32> = self.camera.view_proj();
        let cx = click.0 as f32;
        let cy = click.1 as f32;
        let half_w = w as f32 * 0.5;
        let half_h = h as f32 * 0.5;

        let mut best_dist = 20.0_f32;
        let mut best_label: Option<String> = None;

        for (label, pos) in label_positions {
            let p = nalgebra::Vector4::new(pos[0], pos[1], pos[2], 1.0);
            let clip = view_proj * p;
            if clip.w <= 0.0 {
                continue;
            }

            let ndcx = clip.x / clip.w;
            let ndcy = clip.y / clip.w;

            let sx = (ndcx + 1.0) * half_w;
            let sy = (1.0 - ndcy) * half_h;

            let dx = sx - cx;
            let dy = sy - cy;
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_label = Some(label.clone());
            }
        }

        best_label
    }

    fn active_sheet_index(&self, transition_index: u32) -> Option<usize> {
        if self.dataset.sheets.is_empty() {
            None
        } else {
            Some((transition_index as usize).min(self.dataset.sheets.len() - 1))
        }
    }

    fn frame_label_positions(&self, frame: &lv_data::LisFrame) -> Vec<(String, [f32; 3])> {
        frame
            .labels
            .iter()
            .cloned()
            .zip(frame.instances.iter().map(|inst| inst.position))
            .collect()
    }

    fn render(&mut self, window: &Window) {
        self.app_state.poll_jobs();

        if self.app_state.dataset_changed {
            if let Some(ds) = self.app_state.dataset.clone() {
                self.dataset = ds;
            }
            self.lis_config = self.app_state.lis_config.clone();
            self.rebuild_lis();
            self.app_state.dataset_changed = false;
        }
        if self.app_state.rebuild_lis {
            self.lis_config = self.app_state.lis_config.clone();
            self.rebuild_lis();
            self.app_state.rebuild_lis = false;
        }

        // Sync ego state from GUI app_state
        self.ego_state.cluster_value_min = self.app_state.cluster_min;
        self.ego_state.cluster_value_max = self.app_state.cluster_max;
        self.ego_state.show_secondary = self.app_state.secondary_edges;
        self.ego_state.direction = self.app_state.ego_direction;
        self.ego_state.shared_objects_only = self.app_state.shared_only;
        if !self.app_state.ego_mode {
            self.ego_state.selected = None;
        }

        // advance animation
        let adv = self.timer.tick();
        let total = if self.lis_buffer.streaming {
            self.lis_config.lis_value
        } else {
            self.lis_buffer.frames.len() as u32
        };
        if total > 0 {
            self.slice_index = (self.slice_index + adv.advance_slices) % total;
        }

        // camera uniforms
        let uniforms = ShapeUniforms {
            view_proj: self.camera.view_proj().into(),
            light_dir: [0.577, 0.577, 0.577],
            ambient: 0.15,
        };
        self.ctx
            .queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let (w, h) = self.window_size;
        let edge_uniforms = EdgeUniforms {
            view_proj: self.camera.view_proj().into(),
            viewport_size: [w as f32, h as f32],
            _pad: [0.0; 2],
        };
        self.ctx.queue.write_buffer(
            &self.edge_uniform_buf,
            0,
            bytemuck::bytes_of(&edge_uniforms),
        );

        // get instances
        let frame = if self.lis_buffer.streaming || self.lis_buffer.frames.is_empty() {
            compute_frame(&self.dataset, &self.lis_config, self.slice_index)
        } else {
            let idx = (self.slice_index as usize).min(self.lis_buffer.frames.len() - 1);
            self.lis_buffer.frames[idx].clone()
        };

        // Handle deferred left-click picking
        if let Some(click) = self.last_click_pos.take() {
            if self.app_state.ego_mode {
                let label_positions = self.frame_label_positions(&frame);
                let hit = self.pick_object(click, &label_positions);
                self.ego_state.selected = hit;
            }
        }

        // Build ego-filtered instance list
        let active_sheet_index = self.active_sheet_index(frame.transition_index);
        let visible_set = active_sheet_index
            .and_then(|idx| self.dataset.sheets.get(idx))
            .map(|sheet| compute_visible_objects(sheet, &self.ego_state));

        let has_ego_selection = self.ego_state.selected.is_some() && self.app_state.ego_mode;

        // Apply alpha based on ego visibility
        let filtered_instances: Vec<GpuInstance> = frame
            .instances
            .iter()
            .zip(frame.labels.iter())
            .map(|(inst, label)| {
                let mut gi = *inst;
                if let Some(vs) = &visible_set {
                    if has_ego_selection {
                        if vs.contains(label) {
                            // Selected node: slightly larger + white
                            if self.ego_state.selected.as_deref() == Some(label.as_str()) {
                                gi.size *= 1.05;
                                gi.size_alpha *= 1.05;
                                gi.color = [1.0, 1.0, 1.0, 1.0];
                            }
                            // ego members: full alpha (unchanged)
                        } else {
                            // Non-member: dim alpha
                            gi.color[3] = 0.15;
                        }
                    } else if !vs.contains(label) {
                        // Cluster-value filtered out
                        gi.color[3] = 0.0; // invisible
                    }
                }
                gi
            })
            .collect();

        // Upload per-shape instance data
        for &kind in &self.shape_meshes.kinds {
            let insts: Vec<GpuInstance> = filtered_instances
                .iter()
                .filter(|i| i.shape_id == kind as u32)
                .copied()
                .collect();
            if !insts.is_empty() {
                self.inst_buf
                    .update(kind, &insts, &self.ctx.device, &self.ctx.queue);
            }
        }

        // Build + upload edges
        if has_ego_selection {
            if let (Some(idx), Some(sel)) = (active_sheet_index, &self.ego_state.selected.clone()) {
                let sheet = &self.dataset.sheets[idx];
                let edge_rows = compute_ego_edges(
                    sheet,
                    sel,
                    self.ego_state.show_secondary,
                    self.ego_state.direction,
                );
                let positions: Vec<[f32; 3]> =
                    frame.instances.iter().map(|inst| inst.position).collect();
                let gpu_edges = build_gpu_edges(
                    &edge_rows,
                    &frame.labels,
                    &positions,
                    sel,
                    self.ego_state.show_secondary,
                );
                self.edge_renderer.update(&gpu_edges, &self.ctx.device);
            }
        } else {
            // Clear edges when no selection
            self.edge_renderer.update(&[], &self.ctx.device);
        }

        // ── acquire frame ──────────────────────────────────────────────────────
        let surface_tex = match self
            .ctx
            .surface
            .as_ref()
            .expect("surface required in interactive mode")
            .get_current_texture()
        {
            Ok(t) => t,
            Err(_) => return,
        };
        let surface_view = surface_tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // ── 3-D pass (shapes + edges) ─────────────────────────────────────────
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("3d_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw shapes
            rpass.set_pipeline(&self.pipeline);
            rpass.set_bind_group(0, &self.bind_group, &[]);

            for (i, &kind) in self.shape_meshes.kinds.iter().enumerate() {
                let mesh = &self.shape_meshes.meshes[i];
                let insts: Vec<GpuInstance> = filtered_instances
                    .iter()
                    .filter(|inst| inst.shape_id == kind as u32)
                    .copied()
                    .collect();
                let count = self.inst_buf.instance_count(kind, &insts);
                self.inst_buf.draw(kind, mesh, &mut rpass, count);
            }

            // Draw edges (after shapes, depth write OFF so they overlay)
            rpass.set_pipeline(&self.edge_pipeline);
            rpass.set_bind_group(0, &self.edge_bind_group, &[]);
            self.edge_renderer.draw(&mut rpass);
        }

        // ── egui pass ─────────────────────────────────────────────────────────
        let raw_input = self.egui_winit.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            let _rebuild = self.workspace.show(ctx, &mut self.app_state);
            self.notifications.show(ctx);
        });
        self.egui_winit
            .handle_platform_output(window, full_output.platform_output);

        let clipped = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.ctx.config.width, self.ctx.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.ctx.device, &self.ctx.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            &clipped,
            &screen,
        );

        {
            let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.egui_renderer
                .render(&mut rpass.forget_lifetime(), &clipped, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        surface_tex.present();
    }

    fn on_key(&mut self, code: KeyCode, pressed: bool) {
        let ck = match code {
            KeyCode::ArrowLeft => CameraKey::Left,
            KeyCode::ArrowRight => CameraKey::Right,
            KeyCode::ArrowUp => CameraKey::Up,
            KeyCode::ArrowDown => CameraKey::Down,
            KeyCode::KeyF => CameraKey::ZoomIn,
            KeyCode::KeyS => CameraKey::ZoomOut,
            KeyCode::Backspace => CameraKey::Reset,
            KeyCode::KeyC => CameraKey::Centre,
            _ => return,
        };
        if pressed {
            let (_, action) = self.camera.key_pressed(ck);
            if matches!(action, Some(AppAction::Exit)) {
                // propagated via CloseRequested event
            }
        } else {
            self.camera.key_pressed(ck);
        }
    }
}

// ── winit ApplicationHandler ──────────────────────────────────────────────────

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Lucid Visualization Suite")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 720u32));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        self.renderer = Some(Renderer::new(&window));
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let (Some(renderer), Some(window)) = (&mut self.renderer, &self.window) else {
            return;
        };
        let resp = renderer.egui_winit.on_window_event(window, &event);
        if resp.consumed {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                if let Some(renderer) = &self.renderer {
                    let (w, h) = renderer.window_size;
                    let mut p = renderer.prefs.clone();
                    p.window_width = w;
                    p.window_height = h;
                    if let Err(e) = p.save() {
                        log::warn!("Failed to save prefs: {e:#}");
                    }
                }
                event_loop.exit();
            }
            WindowEvent::Resized(sz) => renderer.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => renderer.render(window),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => renderer.on_key(code, state == ElementState::Pressed),
            WindowEvent::MouseInput { button, state, .. } => {
                let p = renderer.egui_ctx.pointer_latest_pos().unwrap_or_default();
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => {
                        // Store click position; picking runs in render() before
                        // camera drag starts so we defer to render frame.
                        renderer.last_click_pos = Some((p.x as f64, p.y as f64));
                        renderer.camera.mouse_press_left(p.x as f64, p.y as f64);
                    }
                    (MouseButton::Left, ElementState::Released) => {
                        renderer.camera.mouse_release_left();
                    }
                    (MouseButton::Right, ElementState::Pressed) => {
                        renderer.camera.mouse_press_right(p.x as f64, p.y as f64);
                    }
                    (MouseButton::Right, ElementState::Released) => {
                        renderer.camera.mouse_release_right();
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let moved = renderer.camera.mouse_moved(position.x, position.y);
                // If camera actually dragged, cancel the pending pick
                if moved {
                    renderer.last_click_pos = None;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                renderer.camera.scroll(dy);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &winit::event_loop::ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App {
        window: None,
        renderer: None,
    };
    event_loop.run_app(&mut app).expect("run_app");
}
