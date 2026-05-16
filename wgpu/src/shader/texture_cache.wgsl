struct Uniforms {
    transform: mat4x4<f32>,
    bounds: vec4<f32>,  // x, y, w, h in logical pixels
    scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var u_sampler: sampler;
@group(1) @binding(0) var u_texture: texture_2d<f32>;

var<private> uvs: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
);

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = uvs[idx];
    let local = vec2<f32>(
        u.bounds.x + uv.x * u.bounds.z,
        u.bounds.y + uv.y * u.bounds.w,
    );

    var out: VertexOutput;
    out.clip_position = u.transform * vec4<f32>(local * u.scale, 0.0, 1.0);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(u_texture, u_sampler, in.uv);
}
