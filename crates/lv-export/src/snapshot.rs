//! `snapshot` — capture a single frame to an `image::RgbaImage`.
//!
//! Renders entirely off-screen using the existing wgpu device/queue from
//! `WgpuContext`, creating throw-away render and depth textures.
//!
//! Axis overlay: the +X (red), +Y (green) and +Z (blue) world-space axes are
//! drawn as coloured lines with arrowheads in the 3-D pass, then text labels
//! are stamped on in screen-space as a pixel overlay after GPU readback.

use anyhow::{bail, Context as _, Result};
use image::{Rgba, RgbaImage};
use lv_data::{GpuInstance, LisFrame, ShapeKind};
use lv_renderer::{
    pipelines,
    shapes::{cube, cylinder, point, pyramid, sphere, torus, GpuMesh, Lod},
    ArcballCamera, ShapeUniforms, WgpuContext,
};
use wgpu::util::DeviceExt;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_render_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snap_color"),
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
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn make_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snap_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: WgpuContext::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Build one `GpuMesh` per `ShapeKind` (in `ShapeKind::ALL` order).
fn build_all_meshes(device: &wgpu::Device, lod: Lod) -> Vec<GpuMesh> {
    ShapeKind::ALL
        .iter()
        .map(|kind| {
            let (verts, idxs) = match kind {
                ShapeKind::Sphere => sphere::build(lod),
                ShapeKind::Point => point::build(lod),
                ShapeKind::Torus => torus::build(lod),
                ShapeKind::Pyramid => pyramid::build(lod),
                ShapeKind::Cube => cube::build(lod),
                ShapeKind::Cylinder => cylinder::build(lod),
            };
            GpuMesh::upload(device, &format!("snap_{kind}"), &verts, &idxs)
        })
        .collect()
}

// ── axis geometry ─────────────────────────────────────────────────────────────

/// One flat vertex for the axis line pipeline: `[px, py, pz, r, g, b, a]`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AxisVertex {
    pos: [f32; 3],
    color: [f32; 4],
}

impl AxisVertex {
    fn new(pos: [f32; 3], color: [f32; 4]) -> Self {
        Self { pos, color }
    }
}

/// Length of each axis line in world units.
const AXIS_LEN: f32 = 7.0;
/// Length of the arrowhead fins as a fraction of `AXIS_LEN`.
const ARROW_RATIO: f32 = 0.12;
/// How far the arrowhead fin base sits back from the tip.
const ARROW_BASE: f32 = AXIS_LEN * ARROW_RATIO;

/// Build the complete set of axis line vertices (pairs of `AxisVertex`).
///
/// Layout:
///   +X axis — 3 segments (shaft + 2 fins) — red
///   +Y axis — 3 segments — green
///   +Z axis — 3 segments — blue
///
/// Also emits faint dashed negative-half axes for orientation reference.
fn build_axis_vertices() -> Vec<AxisVertex> {
    let mut v: Vec<AxisVertex> = Vec::with_capacity(24);

    struct Axis {
        tip: [f32; 3],
        fin_a: [f32; 3], // first arrowhead fin endpoint
        fin_b: [f32; 3], // second arrowhead fin endpoint
        neg: [f32; 3],   // negative end of shaft (faint)
        color: [f32; 4],
        dim: [f32; 4], // dimmed colour for negative shaft
    }

    let axes = [
        // +X  red
        Axis {
            tip: [AXIS_LEN, 0.0, 0.0],
            fin_a: [AXIS_LEN - ARROW_BASE, ARROW_BASE * 0.5, 0.0],
            fin_b: [AXIS_LEN - ARROW_BASE, -ARROW_BASE * 0.5, 0.0],
            neg: [-AXIS_LEN * 0.3, 0.0, 0.0],
            color: [0.95, 0.22, 0.22, 1.0],
            dim: [0.55, 0.12, 0.12, 0.35],
        },
        // +Y  green
        Axis {
            tip: [0.0, AXIS_LEN, 0.0],
            fin_a: [ARROW_BASE * 0.5, AXIS_LEN - ARROW_BASE, 0.0],
            fin_b: [-ARROW_BASE * 0.5, AXIS_LEN - ARROW_BASE, 0.0],
            neg: [0.0, -AXIS_LEN * 0.3, 0.0],
            color: [0.22, 0.92, 0.38, 1.0],
            dim: [0.12, 0.45, 0.20, 0.35],
        },
        // +Z  blue
        Axis {
            tip: [0.0, 0.0, AXIS_LEN],
            fin_a: [ARROW_BASE * 0.5, 0.0, AXIS_LEN - ARROW_BASE],
            fin_b: [-ARROW_BASE * 0.5, 0.0, AXIS_LEN - ARROW_BASE],
            neg: [0.0, 0.0, -AXIS_LEN * 0.3],
            color: [0.25, 0.55, 0.98, 1.0],
            dim: [0.12, 0.25, 0.55, 0.35],
        },
    ];

    let origin = [0.0f32; 3];

    for ax in &axes {
        // positive shaft: origin → tip
        v.push(AxisVertex::new(origin, ax.color));
        v.push(AxisVertex::new(ax.tip, ax.color));
        // arrowhead fin A
        v.push(AxisVertex::new(ax.tip, ax.color));
        v.push(AxisVertex::new(ax.fin_a, ax.color));
        // arrowhead fin B
        v.push(AxisVertex::new(ax.tip, ax.color));
        v.push(AxisVertex::new(ax.fin_b, ax.color));
        // negative shaft: origin → neg (dimmed)
        v.push(AxisVertex::new(origin, ax.dim));
        v.push(AxisVertex::new(ax.neg, ax.dim));
    }

    v
}

// ── pixel label overlay ───────────────────────────────────────────────────────

/// Project a world-space point through the camera and return the pixel (x, y)
/// in image space (origin top-left).
fn project(world: [f32; 3], view_proj: &[[f32; 4]; 4], w: u32, h: u32) -> Option<(i32, i32)> {
    let vp = nalgebra::Matrix4::from(*view_proj).transpose();
    let p = vp * nalgebra::Vector4::new(world[0], world[1], world[2], 1.0);
    if p.w.abs() < 1e-6 || p.z < 0.0 {
        return None;
    }
    let ndcx = p.x / p.w;
    let ndcy = p.y / p.w;
    // NDC to pixel — note GPU Y goes up, image Y goes down
    let px = ((ndcx + 1.0) * 0.5 * w as f32) as i32;
    let py = ((1.0 - (ndcy + 1.0) * 0.5) * h as f32) as i32;
    if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
        Some((px, py))
    } else {
        // Allow slight overshoot so label stays visible at edge
        Some((px.clamp(-50, w as i32 + 50), py.clamp(-50, h as i32 + 50)))
    }
}

/// Draw a filled rectangle of `color` onto `img`.
fn fill_rect(img: &mut RgbaImage, x: i32, y: i32, ww: i32, hh: i32, color: Rgba<u8>) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    for dy in 0..hh {
        for dx in 0..ww {
            let px = x + dx;
            let py = y + dy;
            if px >= 0 && px < iw && py >= 0 && py < ih {
                img.put_pixel(px as u32, py as u32, color);
            }
        }
    }
}

/// Minimal 5×7 pixel-font bitmap for uppercase A–Z and '+' '-' '0'–'9'.
/// Each char is stored as 7 rows × 5 bits (MSB = left column).
fn char_bitmap(c: char) -> Option<[u8; 7]> {
    // 5-wide, 7-tall bitmaps, MSB = leftmost pixel
    Some(match c {
        'X' => [
            0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001,
        ],
        'Y' => [
            0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000, 0b11111,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        _ => return None,
    })
}

/// Stamp a single character at pixel (x, y) with `scale` (pixel size per bit).
fn draw_char(img: &mut RgbaImage, c: char, x: i32, y: i32, scale: i32, color: Rgba<u8>) {
    if let Some(rows) = char_bitmap(c) {
        for (row, &bits) in rows.iter().enumerate() {
            for col in 0..5i32 {
                let on = (bits >> (4 - col)) & 1 == 1;
                if on {
                    fill_rect(
                        img,
                        x + col * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
    }
}

/// Draw a string (single chars, no wrap) at pixel (x,y) with outline for
/// legibility against any background.
fn draw_label(img: &mut RgbaImage, text: &str, x: i32, y: i32, scale: i32, color: Rgba<u8>) {
    let outline = Rgba([0, 0, 0, 200]);
    let char_w = 5 * scale + scale; // 5 bits + 1 gap
    for (i, c) in text.chars().enumerate() {
        let cx = x + i as i32 * char_w;
        // draw outline (1 px offset in 4 directions)
        for (ox, oy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            draw_char(img, c, cx + ox, y + oy, scale, outline);
        }
        draw_char(img, c, cx, y, scale, color);
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Render a single `LisFrame` to an `RgbaImage` using the provided camera.
///
/// Creates offscreen textures, records a render command, submits to the GPU,
/// and readbacks the pixels.  The Y-axis is flipped to match image conventions.
///
/// Axis lines (+X red, +Y green, +Z blue) and pixel-space labels are overlaid
/// automatically to help viewers understand the spatial orientation of the scene.
pub fn capture_frame(
    ctx: &WgpuContext,
    frame: &LisFrame,
    camera: &ArcballCamera,
    width: u32,
    height: u32,
) -> Result<RgbaImage> {
    if width == 0 || height == 0 {
        bail!("capture_frame: width and height must be > 0");
    }

    let device = &ctx.device;
    let queue = &ctx.queue;

    // ── Offscreen targets ─────────────────────────────────────────────────────
    let color_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (color_tex, color_view) = make_render_target(device, width, height);
    let depth_view = make_depth(device, width, height);

    // ── Build meshes ──────────────────────────────────────────────────────────
    let meshes = build_all_meshes(device, Lod::Mid);

    // ── Shape pipeline ────────────────────────────────────────────────────────
    let bgl = pipelines::uniform_bind_group_layout(device);
    let shape_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("snap_shape_shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../../../assets/shaders/shape_instanced.wgsl").into(),
        ),
    });
    let shape_pipeline = pipelines::create_shape_pipeline_with_format(
        device,
        &shape_shader,
        &bgl,
        color_format,
        WgpuContext::DEPTH_FORMAT,
    );

    // ── Axis pipeline ─────────────────────────────────────────────────────────
    let axis_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("snap_axis_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../../assets/shaders/axis.wgsl").into()),
    });
    let axis_pipeline = pipelines::create_axis_pipeline_with_format(
        device,
        &axis_shader,
        &bgl,
        color_format,
        WgpuContext::DEPTH_FORMAT,
    );

    let axis_verts = build_axis_vertices();
    let axis_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("axis_vbuf"),
        contents: bytemuck::cast_slice(&axis_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // ── Uniforms (shared by both pipelines via same BGL) ──────────────────────
    let mut snap_cam = camera.clone();
    snap_cam.set_aspect(width, height);
    let vp: [[f32; 4]; 4] = snap_cam.view_proj().into();
    let uniforms = ShapeUniforms {
        view_proj: vp,
        light_dir: [0.577_350_26, 0.577_350_26, 0.577_350_26],
        ambient: 0.2,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("snap_uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("snap_bg"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    // ── Encode render pass ────────────────────────────────────────────────────
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("snap_encoder"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("snap_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
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
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // ── Draw shapes ───────────────────────────────────────────────────────
        rpass.set_pipeline(&shape_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);

        for (i, kind) in ShapeKind::ALL.iter().enumerate() {
            let insts: Vec<GpuInstance> = frame
                .instances
                .iter()
                .filter(|inst| inst.shape_id == kind.gpu_id())
                .copied()
                .collect();
            if insts.is_empty() {
                continue;
            }

            let inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("snap_inst"),
                contents: bytemuck::cast_slice::<GpuInstance, u8>(&insts),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let mesh = &meshes[i];
            rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, inst_buf.slice(..));
            rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..mesh.index_count, 0, 0..insts.len() as u32);
        }

        // ── Draw axis lines (same render pass, after shapes) ──────────────────
        rpass.set_pipeline(&axis_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, axis_buf.slice(..));
        rpass.draw(0..axis_verts.len() as u32, 0..1);
    }

    // ── Copy to readable buffer ───────────────────────────────────────────────
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let bytes_per_row_raw = width * 4;
    let bytes_per_row = bytes_per_row_raw.div_ceil(align) * align;
    let buf_size = (bytes_per_row * height) as u64;

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snap_readback"),
        size: buf_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
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

    // ── Map + copy ────────────────────────────────────────────────────────────
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .context("map_async recv")?
        .context("map_async error")?;

    let data = slice.get_mapped_range();
    let bpr = bytes_per_row as usize;
    let bpur = (width * 4) as usize;

    // Flip Y: top of image = bottom of GPU framebuffer
    let mut pixels: Vec<u8> = Vec::with_capacity((width * height * 4) as usize);
    for row in (0..height as usize).rev() {
        pixels.extend_from_slice(&data[row * bpr..row * bpr + bpur]);
    }
    drop(data);
    readback.unmap();

    let mut img = RgbaImage::from_raw(width, height, pixels)
        .context("failed to create RgbaImage from captured pixels")?;

    // ── 2-D pixel label overlay ───────────────────────────────────────────────
    // Project the tip of each axis into screen space and draw a text label
    // slightly beyond it so it doesn't overlap the arrowhead.
    let label_offset = 18i32; // pixels past the projected tip
    let scale = 2i32; // pixel scale for the 5×7 font

    let labels: &[([f32; 3], &str, Rgba<u8>)] = &[
        ([AXIS_LEN * 1.08, 0.0, 0.0], "+X", Rgba([245, 60, 60, 255])),
        ([0.0, AXIS_LEN * 1.08, 0.0], "+Y", Rgba([60, 235, 100, 255])),
        ([0.0, 0.0, AXIS_LEN * 1.08], "+Z", Rgba([70, 145, 250, 255])),
    ];

    for (world, text, color) in labels {
        if let Some((px, py)) = project(*world, &vp, width, height) {
            // Nudge label a few pixels away from the tip toward centre of screen
            let cx = width as i32 / 2;
            let cy = height as i32 / 2;
            let dx = px - cx;
            let dy = py - cy;
            let len = ((dx * dx + dy * dy) as f32).sqrt().max(1.0);
            let ox = (dx as f32 / len * label_offset as f32) as i32;
            let oy = (dy as f32 / len * label_offset as f32) as i32;
            draw_label(&mut img, text, px + ox, py + oy, scale, *color);
        }
    }

    // ── Corner legend ─────────────────────────────────────────────────────────
    // Small fixed-position legend in the bottom-left corner so it is always
    // readable regardless of camera angle.
    let legend = [
        ("+X", Rgba([245, 60, 60, 230u8])),
        ("+Y", Rgba([60, 235, 100, 230u8])),
        ("+Z", Rgba([70, 145, 250, 230u8])),
    ];
    let lx = 14i32;
    let mut ly = height as i32 - 14 - (legend.len() as i32) * (7 * scale + 4);
    for (text, color) in &legend {
        draw_label(&mut img, text, lx, ly, scale, *color);
        ly += 7 * scale + 4;
    }

    Ok(img)
}
