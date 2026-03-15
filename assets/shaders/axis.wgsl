// axis.wgsl
// Simple unlit line shader for axis / grid overlay in Lucid Visualizer.
// Each vertex carries its own world-space position and RGBA colour.
// The pipeline uses LineList topology so every pair of vertices is one segment.

struct Uniforms {
    view_proj : mat4x4<f32>,
    // padded to match the ShapeUniforms layout so we can share one BGL
    light_dir : vec3<f32>,
    ambient   : f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) color    : vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       color    : vec4<f32>,
}

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = uniforms.view_proj * vec4<f32>(v.position, 1.0);
    out.color    = v.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
