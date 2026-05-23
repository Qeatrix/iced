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

// Mitchell-Netravali cubic weight (B = C = 1/3): a smooth reconstruction
// kernel that, sampled at a varying sub-pixel phase, produces far less
// "breathing" of thin high-contrast features than hardware bilinear.
fn cubic_weight(x_in: f32) -> f32 {
    let b = 1.0 / 3.0;
    let c = 1.0 / 3.0;
    let x = abs(x_in);
    let x2 = x * x;
    let x3 = x2 * x;
    if (x < 1.0) {
        return ((12.0 - 9.0 * b - 6.0 * c) * x3
              + (-18.0 + 12.0 * b + 6.0 * c) * x2
              + (6.0 - 2.0 * b)) / 6.0;
    } else if (x < 2.0) {
        return ((-b - 6.0 * c) * x3
              + (6.0 * b + 30.0 * c) * x2
              + (-12.0 * b - 48.0 * c) * x
              + (8.0 * b + 24.0 * c)) / 6.0;
    }
    return 0.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(u_texture));
    // Continuous texel coordinate (texel centers at integers).
    let coord = in.uv * dims - vec2<f32>(0.5);

    // When the cache lands exactly on the device-pixel grid (e.g. the widget
    // snapped a static transform), one texel maps to one device pixel: sample
    // it directly so the result is pixel-perfect crisp. Bicubic would otherwise
    // soften even an aligned image, since Mitchell is an approximating filter.
    let nearest = round(coord);
    let frac = coord - nearest;
    if (max(abs(frac.x), abs(frac.y)) < 0.01) {
        let suv = (nearest + vec2<f32>(0.5)) / dims;
        return textureSampleLevel(u_texture, u_sampler, suv, 0.0);
    }

    // Otherwise (mid-animation, fractional offset) reconstruct with a 4x4
    // Mitchell bicubic to suppress the resampling shimmer of moving edges.
    let base = floor(coord);
    let f = coord - base;
    let wx = vec4<f32>(
        cubic_weight(f.x + 1.0),
        cubic_weight(f.x),
        cubic_weight(f.x - 1.0),
        cubic_weight(f.x - 2.0),
    );
    let wy = vec4<f32>(
        cubic_weight(f.y + 1.0),
        cubic_weight(f.y),
        cubic_weight(f.y - 1.0),
        cubic_weight(f.y - 2.0),
    );

    var color = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let texel = base + vec2<f32>(f32(i) - 1.0, f32(j) - 1.0);
            let suv = (texel + vec2<f32>(0.5)) / dims;
            let w = wx[i] * wy[j];
            color = color + textureSampleLevel(u_texture, u_sampler, suv, 0.0) * w;
            wsum = wsum + w;
        }
    }
    return color / wsum;
}
