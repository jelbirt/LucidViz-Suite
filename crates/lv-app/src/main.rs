//! `lv-app` — Lucid Visualization Suite entry-point.
//!
//! Integrates wgpu rendering (Phase 4) with egui immediate-mode GUI (Phase 5)
//! and the Ego Cluster system (Phase 7).

mod app_state;
mod notifications;
mod prefs;
mod session;

use std::array;
use std::path::Path;
use std::sync::Arc;

use app_state::{build_gpu_edges, compute_ego_edges, compute_visible_objects, EgoClusterState};
use lv_data::{GpuInstance, LisBuffer, LisConfig, LvDataset, ShapeKind};
use lv_gui::state::PlayState;
use lv_gui::{AppState, LucidWorkspace};
use lv_renderer::shapes::{self, cube, cylinder, point, pyramid, sphere, torus};
use lv_renderer::{
    build_lis_buffer, compute_frame, pipelines, AppAction, ArcballCamera, CameraKey, EdgeRenderer,
    EdgeUniforms, FrameTimer, GpuMesh, InstanceBuffer, Lod, ShapeUniforms, WgpuContext,
    EDGE_VERTEX_LAYOUT,
};
use notifications::NotificationQueue;
use prefs::UserPreferences;
use session::{
    delete_session, list_sessions, load_session, rename_session, save_session, AudioSnapshot,
    EgoSnapshot, ExportSnapshot, LisConfigSnapshot, SessionSnapshot,
};

#[cfg(feature = "audio")]
use lv_audio::{BeatMapping, BeatsScheduler, GraduatedConfig};
#[cfg(feature = "export")]
use lv_export::{
    capture_frame, capture_sequence_with_control, ImageFormat, SequenceConfig, SequenceControl,
};
#[cfg(all(feature = "export", feature = "video-export"))]
use lv_export::{export_video_with_control, VideoConfig};

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod demo;

fn make_demo_dataset() -> LvDataset {
    demo::make_demo_dataset()
}

// The 232-line demo dataset function has been moved to demo.rs.

// The following block is compiled-out dead code from the original inline:
#[cfg(any())]
fn _removed() -> LvDataset {
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

    let sheets: Vec<LvSheet> = (0..num_sheets)
        .map(|s| {
            let year_offset = s as f64;

            let rows: Vec<LvRow> = nodes
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
                    LvRow {
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
                        ..LvRow::default()
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

            LvSheet {
                name: format!("{}", 2017 + s),
                sheet_index: s,
                rows,
                edges,
            }
        })
        .collect();

    LvDataset {
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
    dataset: LvDataset,
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
    // Modifier key tracking for keyboard shortcuts
    modifiers: winit::keyboard::ModifiersState,
    // Undo/redo
    undo_stack: lv_gui::UndoStack,
    // Phase 9: preferences + notifications
    prefs: UserPreferences,
    notifications: NotificationQueue,
    #[cfg(feature = "audio")]
    audio_scheduler: BeatsScheduler,
    #[cfg(feature = "audio")]
    last_audio_frame_index: Option<u32>,
}

impl Renderer {
    fn new(window: &Window, prefs: UserPreferences) -> anyhow::Result<Self> {
        let ctx = WgpuContext::new(window)?;

        let dataset = make_demo_dataset();
        let lis_config = lis_config_from_prefs(&prefs);
        let lis_buffer = build_lis_buffer(&dataset, &lis_config);
        let mut app_state = AppState::new();
        app_state.lis_config = lis_config.clone();
        app_state.sync_runtime_snapshot(&dataset, dataset.source_path.clone(), &lis_buffer, 0);
        app_state.session.saved_sessions = list_sessions();

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
                    buffers: &[shapes::Vertex::LAYOUT, pipelines::instance_layout()],
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

        Ok(Self {
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
            app_state,
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
            modifiers: winit::keyboard::ModifiersState::empty(),
            undo_stack: lv_gui::UndoStack::default(),
            prefs,
            notifications: NotificationQueue::default(),
            #[cfg(feature = "audio")]
            audio_scheduler: BeatsScheduler::new(lv_audio::MidiEngine::new()),
            #[cfg(feature = "audio")]
            last_audio_frame_index: None,
        })
    }

    fn handle_audio_request(&mut self) {
        #[cfg(feature = "audio")]
        if let Some(request) = self.app_state.audio.pending_request.take() {
            match request {
                lv_gui::state::AudioRequest::RefreshPorts => {
                    self.app_state.audio.ports = BeatsScheduler::list_ports();
                    self.app_state.audio.status = Some(format!(
                        "Found {} MIDI output port(s).",
                        self.app_state.audio.ports.len()
                    ));
                    if self.app_state.audio.selected_port.is_empty() {
                        if let Some(first) = self.app_state.audio.ports.first() {
                            self.app_state.audio.selected_port = first.clone();
                        }
                    }
                }
                lv_gui::state::AudioRequest::Connect(port) => {
                    match self.audio_scheduler.connect(&port) {
                        Ok(()) => {
                            self.app_state.audio.connected = true;
                            self.app_state.audio.selected_port = port.clone();
                            self.app_state.audio.status = Some(format!("Connected to '{port}'."));
                            self.prefs.audio_port = Some(port);
                        }
                        Err(err) => {
                            self.app_state.audio.connected = false;
                            self.app_state.audio.status =
                                Some(format!("Audio connect failed: {err}"));
                        }
                    }
                }
                lv_gui::state::AudioRequest::Disconnect => {
                    self.audio_scheduler.disconnect();
                    self.app_state.audio.connected = false;
                    self.last_audio_frame_index = None;
                    self.app_state.audio.status = Some("Disconnected MIDI output.".into());
                }
                lv_gui::state::AudioRequest::TestTone => {
                    let result = self.audio_scheduler.test_tone();
                    self.app_state.audio.status = Some(match result {
                        Ok(()) => "Played test tone.".into(),
                        Err(err) => format!("Test tone failed: {err}"),
                    });
                }
            }
        }
    }

    #[cfg(feature = "audio")]
    fn sync_live_audio(&mut self, frame: &lv_data::LisFrame) {
        self.audio_scheduler.beats = self.app_state.audio.beats.max(1);
        self.audio_scheduler.lis_value = self.lis_config.lis_value.max(2);
        self.audio_scheduler.hold_slices = self.app_state.audio.hold_slices.max(1);
        self.audio_scheduler.velocity =
            ((self.app_state.audio.volume.clamp(0.0, 2.0) / 2.0) * 127.0).round() as u8;

        let audio_active = self.app_state.audio.connected
            && self.app_state.audio.live_enabled
            && matches!(self.app_state.play_state, PlayState::Playing);

        if !audio_active {
            if self.last_audio_frame_index.take().is_some() {
                self.audio_scheduler.stop();
            }
            return;
        }

        if self.last_audio_frame_index == Some(frame.slice_index) {
            return;
        }

        if self
            .last_audio_frame_index
            .is_some_and(|last| frame.slice_index < last)
        {
            self.audio_scheduler.stop();
        }

        if let Some(idx) = self.active_sheet_index(frame.transition_index) {
            let grad_cfg = GraduatedConfig {
                semitone_range: self.app_state.audio.semitone_range,
                ..GraduatedConfig::default()
            };
            let mapping = match self.app_state.audio.mapping {
                lv_gui::state::SonificationMapping::CentralityToPitch => {
                    BeatMapping::CentralityToPitch
                }
                lv_gui::state::SonificationMapping::DegreeToVelocity => {
                    BeatMapping::DegreeToVelocity
                }
                lv_gui::state::SonificationMapping::BetweennessPitchClosenessVelocity => {
                    BeatMapping::BetweennessPitchClosenessVelocity
                }
                lv_gui::state::SonificationMapping::ClusterToChannel => {
                    BeatMapping::ClusterToChannel
                }
            };
            self.audio_scheduler.on_frame_advance(
                frame,
                &self.dataset.sheets[idx].rows,
                self.app_state.audio.graduated,
                &grad_cfg,
                mapping,
            );
            self.last_audio_frame_index = Some(frame.slice_index);
        }
    }

    fn handle_session_request(&mut self) {
        if let Some(request) = self.app_state.session.pending_request.take() {
            let result = match request {
                lv_gui::state::SessionRequest::RefreshList => {
                    self.app_state.session.saved_sessions = list_sessions();
                    Ok("Session list refreshed.".to_string())
                }
                lv_gui::state::SessionRequest::Save(name) => {
                    let snapshot = SessionSnapshot {
                        name: name.clone(),
                        source_path: self.app_state.source_path().cloned(),
                        lis_config: LisConfigSnapshot::from(&self.lis_config),
                        slice_index: self.slice_index,
                        cluster_min: self.app_state.cluster.min,
                        cluster_max: self.app_state.cluster.max,
                        ego_mode: self.app_state.cluster.ego_mode,
                        ego: EgoSnapshot::from(&self.ego_state),
                        audio: AudioSnapshot {
                            selected_port: self.app_state.audio.selected_port.clone(),
                            live_enabled: self.app_state.audio.live_enabled,
                            volume: self.app_state.audio.volume,
                            graduated: self.app_state.audio.graduated,
                            semitone_range: self.app_state.audio.semitone_range,
                            beats: self.app_state.audio.beats,
                            hold_slices: self.app_state.audio.hold_slices,
                            mapping: self.app_state.audio.mapping,
                        },
                        export: ExportSnapshot {
                            output_dir: self.app_state.export.output_dir.clone(),
                            filename_prefix: self.app_state.export.filename_prefix.clone(),
                            start_frame: self.app_state.export.start_frame,
                            end_frame: self.app_state.export.end_frame,
                            width: self.app_state.export.width,
                            height: self.app_state.export.height,
                            format: self.app_state.export.format,
                            fps: self.app_state.export.fps,
                            crf: self.app_state.export.crf,
                            codec: self.app_state.export.codec.clone(),
                        },
                    };
                    save_session(&snapshot).map(|_| {
                        self.app_state.session.saved_sessions = list_sessions();
                        self.app_state.session.name = name.clone();
                        format!("Saved session '{name}'.")
                    })
                }
                lv_gui::state::SessionRequest::Delete(name) => delete_session(&name).map(|_| {
                    self.app_state.session.saved_sessions = list_sessions();
                    if self.app_state.session.name == name {
                        self.app_state.session.name.clear();
                    }
                    format!("Deleted session '{name}'.")
                }),
                lv_gui::state::SessionRequest::Rename { from, to } => rename_session(&from, &to)
                    .map(|_| {
                        self.app_state.session.saved_sessions = list_sessions();
                        if self.app_state.session.name == from {
                            self.app_state.session.name = to.clone();
                        }
                        format!("Renamed '{from}' to '{to}'.")
                    }),
                lv_gui::state::SessionRequest::Load(name) => {
                    load_session(&name).and_then(|snapshot| {
                        if let Some(path) = snapshot.source_path.clone() {
                            let dataset = load_dataset_from_path(&path)?;
                            self.dataset = dataset;
                            self.dataset.source_path = Some(path.clone());
                        }
                        self.lis_config = snapshot.lis_config.into();
                        self.app_state.lis_config = self.lis_config.clone();
                        self.slice_index = snapshot.slice_index;
                        self.app_state.pending_slice_index = Some(snapshot.slice_index);
                        self.app_state.cluster.min = snapshot.cluster_min;
                        self.app_state.cluster.max = snapshot.cluster_max;
                        self.app_state.cluster.ego_mode = snapshot.ego_mode;
                        self.ego_state = snapshot.ego.into();
                        self.app_state.cluster.ego_direction = self.ego_state.direction;
                        self.app_state.cluster.secondary_edges = self.ego_state.show_secondary;
                        self.app_state.cluster.shared_only = self.ego_state.shared_objects_only;
                        self.app_state.audio.selected_port = snapshot.audio.selected_port.clone();
                        self.app_state.audio.live_enabled = snapshot.audio.live_enabled;
                        self.app_state.audio.volume = snapshot.audio.volume;
                        self.app_state.audio.graduated = snapshot.audio.graduated;
                        self.app_state.audio.semitone_range = snapshot.audio.semitone_range;
                        self.app_state.audio.beats = snapshot.audio.beats;
                        self.app_state.audio.hold_slices = snapshot.audio.hold_slices;
                        self.app_state.audio.mapping = snapshot.audio.mapping;
                        self.app_state.export.output_dir = snapshot.export.output_dir.clone();
                        self.app_state.export.filename_prefix = snapshot.export.filename_prefix;
                        self.app_state.export.start_frame = snapshot.export.start_frame;
                        self.app_state.export.end_frame = snapshot.export.end_frame;
                        self.app_state.export.width = snapshot.export.width;
                        self.app_state.export.height = snapshot.export.height;
                        self.app_state.export.format = snapshot.export.format;
                        self.app_state.export.fps = snapshot.export.fps;
                        self.app_state.export.crf = snapshot.export.crf;
                        self.app_state.export.codec = snapshot.export.codec;
                        #[cfg(feature = "audio")]
                        if self.app_state.audio.selected_port.is_empty() {
                            self.audio_scheduler.disconnect();
                            self.app_state.audio.connected = false;
                            self.last_audio_frame_index = None;
                        } else {
                            match self
                                .audio_scheduler
                                .connect(&self.app_state.audio.selected_port)
                            {
                                Ok(()) => {
                                    self.app_state.audio.connected = true;
                                }
                                Err(err) => {
                                    self.app_state.audio.connected = false;
                                    self.app_state.audio.status =
                                        Some(format!("Session audio reconnect failed: {err}"));
                                }
                            }
                        }
                        self.rebuild_lis();
                        self.app_state.session.name = name.clone();
                        Ok(format!("Loaded session '{name}'."))
                    })
                }
            };
            self.app_state.session.loading = false;
            if result.is_err() {
                self.app_state.session.confirm_delete = None;
                self.app_state.session.renaming = None;
            }
            self.app_state.session.status = Some(match result {
                Ok(msg) => msg,
                Err(err) => format!("Session error: {err:#}"),
            });
        }
    }

    #[cfg(feature = "export")]
    fn handle_export_request(&mut self, frame: &lv_data::LisFrame) {
        if let Some(request) = self.app_state.export.pending_request.take() {
            if self.app_state.export.job.is_some() {
                self.app_state.export.status = Some("An export is already running.".into());
                return;
            }
            match request.kind {
                lv_gui::state::ExportKind::CurrentFrame => {
                    let prefix = if request.filename_prefix.trim().is_empty() {
                        "frame"
                    } else {
                        request.filename_prefix.trim()
                    };
                    let extension = match request.format {
                        lv_gui::state::ExportImageFormat::Png => "png",
                        lv_gui::state::ExportImageFormat::Tga => "tga",
                    };
                    let path = request
                        .output_dir
                        .join(format!("{prefix}_{:06}.{extension}", self.slice_index));
                    let result = std::fs::create_dir_all(&request.output_dir)
                        .map_err(anyhow::Error::from)
                        .and_then(|_| {
                            capture_frame(
                                &self.ctx,
                                frame,
                                &self.camera,
                                request.width,
                                request.height,
                            )
                        })
                        .and_then(|img| img.save(&path).map_err(anyhow::Error::from));
                    match result {
                        Ok(()) => {
                            self.app_state.export.status =
                                Some(format!("Saved current frame to {}.", path.display()));
                            self.notifications
                                .push(notifications::Notification::info(format!(
                                    "Exported {}",
                                    path.display()
                                )));
                        }
                        Err(err) => {
                            self.app_state.export.status = Some(format!("Export failed: {err:#}"));
                        }
                    }
                }
                lv_gui::state::ExportKind::Sequence => {
                    self.start_export_job(request, false);
                }
                lv_gui::state::ExportKind::Video => {
                    self.start_export_job(request, true);
                }
            }
        }
    }

    #[cfg(feature = "export")]
    fn start_export_job(&mut self, request: lv_gui::state::ExportRequest, video: bool) {
        use std::sync::atomic::AtomicBool;
        use std::sync::mpsc;
        use std::thread;

        let dataset = self.dataset.clone();
        let lis_config = self.lis_config.clone();
        let lis_buffer = self.lis_buffer.clone();
        let camera = self.camera.clone();
        let (progress_tx, progress_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let cancel_for_thread = Arc::clone(&cancel_flag);
        let kind = request.kind;

        thread::spawn(move || {
            let result = lv_renderer::WgpuContext::new_headless()
                .and_then(|ctx| {
                    let image_format = match request.format {
                        lv_gui::state::ExportImageFormat::Png => ImageFormat::Png,
                        lv_gui::state::ExportImageFormat::Tga => ImageFormat::Tga,
                    };
                    if video {
                        #[cfg(feature = "video-export")]
                        {
                            let output_name = if request.filename_prefix.trim().is_empty() {
                                "export.mp4".to_string()
                            } else {
                                format!("{}.mp4", request.filename_prefix.trim())
                            };
                            let output_path = request.output_dir.join(output_name);
                            let video_config = VideoConfig {
                                output_path: output_path.clone(),
                                fps: request.fps,
                                crf: request.crf,
                                codec: request.codec.clone(),
                            };
                            export_video_with_control(
                                &ctx,
                                &dataset,
                                &lis_config,
                                &lis_buffer,
                                &camera,
                                request.width,
                                request.height,
                                request.start_frame,
                                request.end_frame,
                                &video_config,
                                &progress_tx,
                                Some(cancel_for_thread.as_ref()),
                            )
                            .map(|_| format!("Saved video export to {}.", output_path.display()))
                        }
                        #[cfg(not(feature = "video-export"))]
                        {
                            Err(anyhow::anyhow!("video export not compiled in"))
                        }
                    } else {
                        let seq_config = SequenceConfig {
                            output_dir: request.output_dir.clone(),
                            filename_prefix: if request.filename_prefix.trim().is_empty() {
                                "frame".into()
                            } else {
                                request.filename_prefix.trim().to_string()
                            },
                            start_frame: request.start_frame,
                            end_frame: request.end_frame,
                            width: request.width,
                            height: request.height,
                            format: image_format,
                            overwrite: true,
                        };
                        capture_sequence_with_control(
                            &ctx,
                            &dataset,
                            &lis_config,
                            &lis_buffer,
                            &camera,
                            &seq_config,
                            SequenceControl {
                                progress: &progress_tx,
                                cancel: Some(cancel_for_thread.as_ref()),
                            },
                        )
                        .map(|_| {
                            format!(
                                "Saved image sequence to {}.",
                                seq_config.output_dir.display()
                            )
                        })
                    }
                })
                .map_err(|err| err.to_string());
            let _ = result_tx.send(result);
        });

        self.app_state.export.status = Some(match kind {
            lv_gui::state::ExportKind::Sequence => "Started image-sequence export.".into(),
            lv_gui::state::ExportKind::Video => "Started video export.".into(),
            lv_gui::state::ExportKind::CurrentFrame => unreachable!(),
        });
        self.app_state.export.job = Some(lv_gui::state::ExportJob {
            kind,
            progress: 0.0,
            progress_rx,
            result_rx,
            cancel_flag,
        });
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
        let source_path = self
            .dataset
            .source_path
            .clone()
            .or_else(|| self.app_state.source_path().cloned());
        self.app_state
            .sync_runtime_snapshot(&self.dataset, source_path, &self.lis_buffer, 0);
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
        self.app_state.poll_export_job();
        self.handle_audio_request();
        self.handle_session_request();

        // Push undo snapshot before processing state-changing events.
        if self.app_state.pending_dataset_load.is_some() || self.app_state.rebuild_lis {
            self.undo_stack.push(self.app_state.snapshot());
        }

        if let Some(pending) = self.app_state.pending_dataset_load.take() {
            self.dataset = pending.dataset;
            self.dataset.source_path = Some(pending.source_path.clone());
            if let Some(path) = self.dataset.source_path.clone() {
                self.prefs.push_recent(path);
                if let Err(err) = self.prefs.save() {
                    log::warn!("Failed to save prefs after dataset load: {err:#}");
                }
            }
            self.lis_config = self.app_state.lis_config.clone();
            self.rebuild_lis();
        }
        if self.app_state.rebuild_lis {
            self.lis_config = self.app_state.lis_config.clone();
            self.rebuild_lis();
            self.app_state.rebuild_lis = false;
        }

        self.lis_config = self.app_state.lis_config.clone();
        self.timer.set_target_fps(self.lis_config.target_fps);
        self.timer.set_speed(self.lis_config.speed);

        // Sync ego state from GUI app_state
        self.ego_state.cluster_value_min = self.app_state.cluster.min;
        self.ego_state.cluster_value_max = self.app_state.cluster.max;
        self.ego_state.show_secondary = self.app_state.cluster.secondary_edges;
        self.ego_state.direction = self.app_state.cluster.ego_direction;
        self.ego_state.shared_objects_only = self.app_state.cluster.shared_only;
        if !self.app_state.cluster.ego_mode {
            self.ego_state.selected = None;
        }

        let total = if self.lis_buffer.streaming {
            self.lis_config.lis_value
        } else {
            self.lis_buffer.frames.len() as u32
        };
        if total > 0 {
            if let Some(requested) = self.app_state.pending_slice_index.take() {
                self.slice_index = requested.min(total - 1);
            } else {
                self.slice_index = self.app_state.slice_index().min(total - 1);
            }
            let adv = self.timer.tick();
            self.slice_index = advance_slice_index(
                self.slice_index,
                total,
                self.app_state.play_state,
                self.lis_config.looping,
                adv.advance_slices,
            );
        }
        self.app_state.sync_runtime_snapshot(
            &self.dataset,
            self.dataset.source_path.clone(),
            &self.lis_buffer,
            self.slice_index,
        );

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

        #[cfg(feature = "audio")]
        self.sync_live_audio(&frame);

        #[cfg(feature = "export")]
        self.handle_export_request(&frame);

        // Handle deferred left-click picking
        if let Some(click) = self.last_click_pos.take() {
            if self.app_state.cluster.ego_mode {
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

        let has_ego_selection =
            self.ego_state.selected.is_some() && self.app_state.cluster.ego_mode;

        // Apply alpha based on ego visibility
        let filtered_instances: Vec<GpuInstance> = frame
            .instances
            .iter()
            .zip(frame.labels.iter())
            .map(|(inst, label)| {
                let mut gi = apply_object_override(*inst, self.app_state.overrides.get(label));
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

        let grouped_instances = group_instances_by_shape(&filtered_instances);

        // Upload per-shape instance data
        for &kind in &self.shape_meshes.kinds {
            self.inst_buf.update(
                kind,
                &grouped_instances[kind as usize],
                &self.ctx.device,
                &self.ctx.queue,
            );
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
                self.edge_renderer
                    .update(&gpu_edges, &self.ctx.device, &self.ctx.queue);
            }
        } else {
            // Clear edges when no selection
            self.edge_renderer
                .update(&[], &self.ctx.device, &self.ctx.queue);
        }

        // ── acquire frame ──────────────────────────────────────────────────────
        let surface_tex = match self
            .ctx
            .surface
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("surface required in interactive mode"))
            .map_err(|err| {
                log::error!("{err:#}");
                err
            })
            .ok()
            .and_then(|surface| surface.get_current_texture().ok())
        {
            Some(t) => t,
            None => return,
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
                let count = self.inst_buf.instance_count(kind);
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
        if !pressed {
            return;
        }

        let ctrl = self.modifiers.control_key();
        let shift = self.modifiers.shift_key();

        // ── Application shortcuts ─────────────────────────────────────
        match (ctrl, shift, code) {
            (false, false, KeyCode::Space) => {
                // Play/pause toggle
                self.app_state.play_state = match self.app_state.play_state {
                    PlayState::Playing => PlayState::Paused,
                    PlayState::Paused | PlayState::Stopped => PlayState::Playing,
                };
                return;
            }
            (true, false, KeyCode::KeyS) => {
                // Save session
                let name = self.app_state.session.name.trim().to_string();
                if !name.is_empty() {
                    self.app_state.session.pending_request =
                        Some(lv_gui::state::SessionRequest::Save(name));
                }
                return;
            }
            (false, false, KeyCode::KeyN) => {
                // Next slice
                let max = self.dataset.sheets.len().saturating_sub(1) as u32;
                if self.slice_index < max {
                    self.app_state.pending_slice_index = Some(self.slice_index + 1);
                }
                return;
            }
            (false, false, KeyCode::KeyP) => {
                // Previous slice
                if self.slice_index > 0 {
                    self.app_state.pending_slice_index = Some(self.slice_index.saturating_sub(1));
                }
                return;
            }
            (true, false, KeyCode::KeyZ) => {
                // Undo
                if let Some(snap) = self.undo_stack.undo() {
                    let snap = snap.clone();
                    self.app_state.restore_snapshot(&snap);
                }
                return;
            }
            (true, true, KeyCode::KeyZ) => {
                // Redo
                if let Some(snap) = self.undo_stack.redo() {
                    let snap = snap.clone();
                    self.app_state.restore_snapshot(&snap);
                }
                return;
            }
            _ => {}
        }

        // ── Camera keys ───────────────────────────────────────────────
        let ck = match code {
            KeyCode::ArrowLeft => CameraKey::Left,
            KeyCode::ArrowRight => CameraKey::Right,
            KeyCode::ArrowUp => CameraKey::Up,
            KeyCode::ArrowDown => CameraKey::Down,
            KeyCode::KeyF => CameraKey::ZoomIn,
            KeyCode::Backspace => CameraKey::Reset,
            KeyCode::KeyC => CameraKey::Centre,
            // KeyS is now save-session shortcut (Ctrl+S) or still zoom-out if
            // ctrl is not pressed (but N/P took priority); keep zoom-out for
            // the plain S case only when not caught above.
            KeyCode::KeyS if !ctrl => CameraKey::ZoomOut,
            _ => return,
        };
        let (_, action) = self.camera.key_pressed(ck);
        if matches!(action, Some(AppAction::Exit)) {
            // propagated via CloseRequested event
        }
    }
}

fn advance_slice_index(
    current: u32,
    total: u32,
    play_state: PlayState,
    looping: bool,
    advance_slices: u32,
) -> u32 {
    if total == 0 {
        return 0;
    }
    match play_state {
        PlayState::Playing => {
            if looping {
                current.saturating_add(advance_slices) % total
            } else {
                current.saturating_add(advance_slices).min(total - 1)
            }
        }
        PlayState::Paused => current.min(total - 1),
        PlayState::Stopped => 0,
    }
}

fn apply_object_override(
    mut instance: GpuInstance,
    override_cfg: Option<&lv_gui::state::ObjectOverride>,
) -> GpuInstance {
    if let Some(override_cfg) = override_cfg {
        if let Some(shape) = override_cfg.shape {
            instance.shape_id = shape.gpu_id();
        }
        if let Some(size) = override_cfg.size {
            instance.size = size as f32;
        }
        if let Some(color) = override_cfg.color {
            instance.color[0] = color[0];
            instance.color[1] = color[1];
            instance.color[2] = color[2];
        }
    }
    instance
}

fn group_instances_by_shape(instances: &[GpuInstance]) -> [Vec<GpuInstance>; ShapeKind::ALL.len()] {
    let mut grouped: [Vec<GpuInstance>; ShapeKind::ALL.len()] = array::from_fn(|_| Vec::new());
    for instance in instances {
        let shape_idx = (instance.shape_id as usize).min(ShapeKind::ALL.len() - 1);
        grouped[shape_idx].push(*instance);
    }
    grouped
}

fn load_dataset_from_path(path: &Path) -> anyhow::Result<LvDataset> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => Ok(lv_data::load_dataset_json(path)?),
        _ => Ok(lv_data::read_lv_xlsx(path)?),
    }
}

fn lis_config_from_prefs(prefs: &UserPreferences) -> LisConfig {
    LisConfig {
        lis_value: prefs.default_lis.max(2),
        target_fps: prefs.default_fps,
        ..LisConfig::default()
    }
}

#[cfg(test)]
mod runtime_tests {
    use super::{
        advance_slice_index, apply_object_override, group_instances_by_shape, lis_config_from_prefs,
    };
    use lv_data::{GpuInstance, ShapeKind};
    use lv_gui::state::{ObjectOverride, PlayState};
    use std::path::PathBuf;

    use crate::prefs::UserPreferences;

    #[test]
    fn advance_slice_index_respects_play_state_and_looping() {
        assert_eq!(advance_slice_index(3, 10, PlayState::Playing, true, 2), 5);
        assert_eq!(advance_slice_index(9, 10, PlayState::Playing, true, 2), 1);
        assert_eq!(advance_slice_index(9, 10, PlayState::Playing, false, 2), 9);
        assert_eq!(advance_slice_index(4, 10, PlayState::Paused, true, 5), 4);
        assert_eq!(advance_slice_index(4, 10, PlayState::Stopped, true, 5), 0);
    }

    #[test]
    fn apply_object_override_updates_shape_size_and_rgb_only() {
        let instance = GpuInstance {
            position: [0.0, 0.0, 0.0],
            size: 1.0,
            size_alpha: 0.5,
            _pad0: [0.0; 3],
            color: [0.1, 0.2, 0.3, 0.8],
            spin: [0.0, 0.0, 0.0],
            shape_id: ShapeKind::Sphere.gpu_id(),
        };
        let override_cfg = ObjectOverride {
            shape: Some(ShapeKind::Cube),
            color: Some([0.9, 0.8, 0.7]),
            size: Some(2.5),
        };

        let updated = apply_object_override(instance, Some(&override_cfg));

        assert_eq!(updated.shape_id, ShapeKind::Cube.gpu_id());
        assert_eq!(updated.size, 2.5);
        assert_eq!(updated.color, [0.9, 0.8, 0.7, 0.8]);
        assert_eq!(updated.size_alpha, 0.5);
    }

    #[test]
    fn lis_config_from_prefs_applies_defaults_safely() {
        let prefs = UserPreferences {
            recent_files: vec![PathBuf::from("/tmp/data.xlsx")],
            window_width: 1600,
            window_height: 900,
            default_lis: 1,
            default_fps: Some(24),
            audio_port: None,
            brandes_threads: None,
        };

        let config = lis_config_from_prefs(&prefs);

        assert_eq!(config.lis_value, 2);
        assert_eq!(config.target_fps, Some(24));
        assert!(config.looping);
        assert_eq!(config.speed, 1.0);
    }

    #[test]
    fn group_instances_by_shape_batches_without_re_filtering() {
        let sphere = GpuInstance {
            position: [0.0, 0.0, 0.0],
            size: 1.0,
            size_alpha: 0.0,
            _pad0: [0.0; 3],
            color: [1.0, 0.0, 0.0, 1.0],
            spin: [0.0, 0.0, 0.0],
            shape_id: ShapeKind::Sphere.gpu_id(),
        };
        let cube = GpuInstance {
            shape_id: ShapeKind::Cube.gpu_id(),
            ..sphere
        };

        let grouped = group_instances_by_shape(&[sphere, cube, sphere]);

        assert_eq!(grouped[ShapeKind::Sphere as usize].len(), 2);
        assert_eq!(grouped[ShapeKind::Cube as usize].len(), 1);
        assert!(grouped[ShapeKind::Torus as usize].is_empty());
    }
}

// ── winit ApplicationHandler ──────────────────────────────────────────────────

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    prefs: UserPreferences,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Lucid Visualization Suite")
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.prefs.window_width,
                self.prefs.window_height,
            ));
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("Failed to create window: {err:#}");
                event_loop.exit();
                return;
            }
        };
        let renderer = match Renderer::new(&window, self.prefs.clone()) {
            Ok(renderer) => renderer,
            Err(err) => {
                log::error!("Failed to initialize renderer: {err:#}");
                event_loop.exit();
                return;
            }
        };
        self.renderer = Some(renderer);
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
            WindowEvent::ModifiersChanged(mods) => {
                renderer.modifiers = mods.state();
            }
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

    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--headless") {
        run_headless(&args);
        return;
    }

    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            log::error!("Failed to create event loop: {err:#}");
            return;
        }
    };
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App {
        window: None,
        renderer: None,
        prefs: UserPreferences::load(),
    };
    if let Err(err) = event_loop.run_app(&mut app) {
        log::error!("Application event loop failed: {err:#}");
    }
}

/// Headless mode: load dataset, run MF -> AS pipeline, write JSON coordinates.
///
/// Usage: lv-app --headless --input <path> --output <path>
fn run_headless(args: &[String]) {
    let input_path = args
        .windows(2)
        .find(|w| w[0] == "--input")
        .map(|w| std::path::PathBuf::from(&w[1]));
    let output_path = args
        .windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| std::path::PathBuf::from(&w[1]));

    let Some(input_path) = input_path else {
        eprintln!("--headless requires --input <path>");
        std::process::exit(1);
    };
    let Some(output_path) = output_path else {
        eprintln!("--headless requires --output <path>");
        std::process::exit(1);
    };

    eprintln!("Headless mode: loading {}", input_path.display());
    let dataset = match load_dataset_from_path(&input_path) {
        Ok(ds) => ds,
        Err(err) => {
            eprintln!("Failed to load dataset: {err:#}");
            std::process::exit(1);
        }
    };

    use as_pipeline::pipeline::run_pipeline;
    use as_pipeline::types::{
        AsPipelineInput, CentralityMode, MdsConfig, MdsDimMode, NormalizationMode, ProcrustesMode,
    };
    use std::collections::BTreeSet;

    let all_labels: Vec<String> = {
        let mut set = BTreeSet::new();
        for sheet in &dataset.sheets {
            for row in &sheet.rows {
                set.insert(row.label.clone());
            }
        }
        set.into_iter().collect()
    };

    let datasets: Vec<(String, ndarray::Array2<f64>)> = dataset
        .sheets
        .iter()
        .map(|sheet| {
            let n = all_labels.len();
            let mut adj = ndarray::Array2::<f64>::zeros((n, n));
            let label_idx: std::collections::HashMap<&str, usize> = all_labels
                .iter()
                .enumerate()
                .map(|(i, l)| (l.as_str(), i))
                .collect();
            for edge in &sheet.edges {
                if let (Some(&i), Some(&j)) = (
                    label_idx.get(edge.from.as_str()),
                    label_idx.get(edge.to.as_str()),
                ) {
                    adj[(i, j)] = edge.strength;
                }
            }
            (sheet.name.clone(), adj)
        })
        .collect();

    let input = AsPipelineInput {
        datasets,
        labels: all_labels,
        mds_config: MdsConfig::Auto,
        procrustes_mode: ProcrustesMode::None,
        mds_dims: MdsDimMode::Fixed(3),
        normalize: true,
        normalization_mode: NormalizationMode::Independent,
        target_range: 500.0,
        procrustes_scale: false,
        centrality_mode: CentralityMode::Directed,
    };

    eprintln!("Running AS pipeline...");
    let result = match run_pipeline(&input) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("Pipeline failed: {err:#}");
            std::process::exit(1);
        }
    };

    let output: Vec<serde_json::Value> = result
        .coordinates
        .iter()
        .enumerate()
        .map(|(step, coords)| {
            let points: Vec<serde_json::Value> = coords
                .labels
                .iter()
                .enumerate()
                .map(|(i, label)| {
                    let row = coords.row(i);
                    serde_json::json!({
                        "label": label,
                        "x": row.first().copied().unwrap_or(0.0),
                        "y": row.get(1).copied().unwrap_or(0.0),
                        "z": row.get(2).copied().unwrap_or(0.0),
                    })
                })
                .collect();
            serde_json::json!({
                "step": step,
                "stress": coords.stress,
                "coordinates": points,
            })
        })
        .collect();

    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    if let Err(err) = std::fs::write(&output_path, &json) {
        eprintln!("Failed to write output: {err}");
        std::process::exit(1);
    }
    eprintln!(
        "Wrote {} steps to {}",
        result.coordinates.len(),
        output_path.display()
    );
}
