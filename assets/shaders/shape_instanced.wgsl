// shape_instanced.wgsl
// Instanced shape vertex + fragment shader for Lucid Visualization Suite.
// See implementation_plan.md §4.2 for full GpuInstance layout (64 bytes).

// ─── Uniforms ────────────────────────────────────────────────────────────────

struct Uniforms {
    view_proj  : mat4x4<f32>,
    light_dir  : vec3<f32>,
    ambient    : f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Vertex input (per-vertex) ───────────────────────────────────────────────

struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
}

// ─── Instance input (per-instance) ──────────────────────────────────────────
// Matches GpuInstance repr(C) layout exactly:
//   offset  0: position   vec3<f32>   (locations 2)
//   offset 12: size       f32         (location 3)
//   offset 16: size_alpha f32         (location 4)
//   offset 20: _pad0      vec3<f32>   (location 5)  ← padding, unused in shader
//   offset 32: color      vec4<f32>   (location 6)
//   offset 48: spin       vec3<f32>   (location 7)
//   offset 60: shape_id   u32         (location 8)

struct InstanceInput {
    @location(2) inst_position   : vec3<f32>,
    @location(3) inst_size       : f32,
    @location(4) inst_size_alpha : f32,
    @location(5) inst_pad0       : vec3<f32>,   // padding — not used
    @location(6) inst_color      : vec4<f32>,
    @location(7) inst_spin       : vec3<f32>,
    @location(8) inst_shape_id   : u32,
}

// ─── Vertex output ───────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       world_normal : vec3<f32>,
    @location(1)       color        : vec4<f32>,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build rotation matrix from Euler angles (radians): Rz * Ry * Rx
fn euler_rotation(spin_rad: vec3<f32>) -> mat3x3<f32> {
    let sx = sin(spin_rad.x); let cx = cos(spin_rad.x);
    let sy = sin(spin_rad.y); let cy = cos(spin_rad.y);
    let sz = sin(spin_rad.z); let cz = cos(spin_rad.z);

    // Rx
    let rx = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0,  cx,  sx),
        vec3<f32>(0.0, -sx,  cx),
    );
    // Ry
    let ry = mat3x3<f32>(
        vec3<f32>( cy, 0.0, -sy),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>( sy, 0.0,  cy),
    );
    // Rz
    let rz = mat3x3<f32>(
        vec3<f32>( cz,  sz, 0.0),
        vec3<f32>(-sz,  cz, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    return rz * ry * rx;
}

// ─── Vertex shader ───────────────────────────────────────────────────────────

@vertex
fn vs_main(vert: VertexInput, inst: InstanceInput) -> VertexOutput {
    let DEG_TO_RAD = 3.14159265358979323846 / 180.0;

    // Apply spin rotation to mesh vertex
    let spin_rad = inst.inst_spin * DEG_TO_RAD;
    let rot      = euler_rotation(spin_rad);
    let rotated  = rot * vert.position;

    // Scale and translate into world space
    let world_pos = vec4<f32>(rotated * inst.inst_size + inst.inst_position, 1.0);

    var out: VertexOutput;
    out.clip_pos    = uniforms.view_proj * world_pos;
    out.world_normal = normalize(rot * vert.normal);
    out.color        = inst.inst_color;
    return out;
}

// ─── Fragment shader (Blinn-Phong) ───────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n         = normalize(in.world_normal);
    let l         = normalize(uniforms.light_dir);
    let diffuse   = max(dot(n, l), 0.0);
    let intensity = uniforms.ambient + (1.0 - uniforms.ambient) * diffuse;
    return vec4<f32>(in.color.rgb * intensity, in.color.a);
}
