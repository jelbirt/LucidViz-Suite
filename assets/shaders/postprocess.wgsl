// postprocess.wgsl — FXAA (Fast Approximate Anti-Aliasing) fullscreen pass.
//
// Simplified FXAA 3.11 based on Timothy Lottes (NVIDIA).
// Operates on the scene color texture to smooth aliased edges.

@group(0) @binding(0)
var scene_tex: texture_2d<f32>;
@group(0) @binding(1)
var scene_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle — 3 vertices, no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    // CCW triangle covering the entire clip space:
    //   idx=0 → (-1, -1)   idx=1 → (3, -1)   idx=2 → (-1, 3)
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Perceived luminance (Rec. 709).
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(scene_tex));
    let texel = 1.0 / dims;

    // Sample center and 4 neighbours.
    let center = textureSample(scene_tex, scene_sampler, in.uv);
    let n  = textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(0.0, -texel.y));
    let s  = textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(0.0,  texel.y));
    let e  = textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>( texel.x, 0.0));
    let w  = textureSample(scene_tex, scene_sampler, in.uv + vec2<f32>(-texel.x, 0.0));

    let luma_c = luma(center.rgb);
    let luma_n = luma(n.rgb);
    let luma_s = luma(s.rgb);
    let luma_e = luma(e.rgb);
    let luma_w = luma(w.rgb);

    let luma_min = min(luma_c, min(min(luma_n, luma_s), min(luma_e, luma_w)));
    let luma_max = max(luma_c, max(max(luma_n, luma_s), max(luma_e, luma_w)));
    let luma_range = luma_max - luma_min;

    // Early exit: no edge detected.
    let FXAA_EDGE_THRESHOLD: f32 = 0.125;
    let FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0625;
    if luma_range < max(FXAA_EDGE_THRESHOLD_MIN, luma_max * FXAA_EDGE_THRESHOLD) {
        return center;
    }

    // Sub-pixel aliasing test.
    let luma_avg = (luma_n + luma_s + luma_e + luma_w) * 0.25;
    let sub_pixel = clamp(abs(luma_avg - luma_c) / luma_range, 0.0, 1.0);
    let sub_pixel_blend = smoothstep(0.0, 1.0, sub_pixel);
    let blend = sub_pixel_blend * sub_pixel_blend * 0.75;

    // Determine edge direction.
    let horiz = abs(luma_n + luma_s - 2.0 * luma_c);
    let vert  = abs(luma_e + luma_w - 2.0 * luma_c);
    let is_horizontal = horiz >= vert;

    // Blend along the edge normal.
    var offset: vec2<f32>;
    if is_horizontal {
        offset = vec2<f32>(0.0, texel.y);
    } else {
        offset = vec2<f32>(texel.x, 0.0);
    }

    let pos_sample = textureSample(scene_tex, scene_sampler, in.uv + offset * blend);
    let neg_sample = textureSample(scene_tex, scene_sampler, in.uv - offset * blend);

    return mix(center, (pos_sample + neg_sample) * 0.5, vec4<f32>(blend));
}
