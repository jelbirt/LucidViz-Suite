// edge.wgsl
// Edge / line rendering shader for Lucid Visualization Suite.
// Expands each edge into a screen-space quad (~2 px wide).

// ─── Uniforms ────────────────────────────────────────────────────────────────

struct Uniforms {
    view_proj       : mat4x4<f32>,
    viewport_size   : vec2<f32>,
    _pad            : vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Per-vertex ───────────────────────────────────────────────────────────────
// Each logical edge is drawn as 4 vertices (triangle-strip quad).
// The CPU encodes both endpoints in every vertex and uses a side flag.

struct VertexInput {
    @location(0) from_pos  : vec3<f32>,
    @location(1) to_pos    : vec3<f32>,
    @location(2) color     : vec4<f32>,
    @location(3) side      : f32,   // -1.0 or +1.0 — which side of the quad
}

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       color    : vec4<f32>,
}

// ─── Vertex shader ────────────────────────────────────────────────────────────

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let LINE_WIDTH_PX = 2.0;

    // Project both endpoints to clip space
    let clip_from = uniforms.view_proj * vec4<f32>(in.from_pos, 1.0);
    let clip_to   = uniforms.view_proj * vec4<f32>(in.to_pos,   1.0);

    // Guard against endpoints behind the near plane (w <= 0) which would
    // produce NaN from the perspective divide and corrupt the quad.
    let safe_w_from = max(clip_from.w, 0.0001);
    let safe_w_to   = max(clip_to.w,   0.0001);

    // NDC positions (safe from division by zero/negative)
    let ndc_from = clip_from.xy / safe_w_from;
    let ndc_to   = clip_to.xy   / safe_w_to;

    // Screen-space direction (pixels). Guard against zero-length edges
    // (coincident endpoints or both clamped) which would NaN in normalize.
    let raw_dir = (ndc_to - ndc_from) * uniforms.viewport_size * 0.5;
    let dir_len = length(raw_dir);
    let dir_px  = select(vec2<f32>(1.0, 0.0), raw_dir / dir_len, dir_len > 1e-5);
    // Perpendicular in NDC
    let perp_ndc = vec2<f32>(-dir_px.y, dir_px.x) / (uniforms.viewport_size * 0.5) * LINE_WIDTH_PX;

    // Choose which endpoint this vertex belongs to (side encodes both endpoint
    // and quad side: +1 = to+perp, -1 = to-perp, +2 = from+perp, -2 = from-perp)
    var base_clip: vec4<f32>;
    var s: f32;
    if in.side > 1.5 {
        base_clip = clip_to; s = 1.0;
    } else if in.side > 0.5 {
        base_clip = clip_to; s = -1.0;
    } else if in.side > -0.5 {
        base_clip = clip_from; s = 1.0;
    } else {
        base_clip = clip_from; s = -1.0;
    }

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(base_clip.xy + perp_ndc * s * base_clip.w, base_clip.zw);
    out.color    = in.color;
    return out;
}

// ─── Fragment shader ─────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
