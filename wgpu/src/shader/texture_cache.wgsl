struct Uniforms {
    transform: mat4x4<f32>,
    bounds: vec4<f32>,  // x, y, w, h in logical pixels
    // [uv_max.x, uv_max.y, scale, _pad]. `uv_max` is the fraction of the
    // texture dimensions that actually holds content (the texture may be
    // over-allocated to amortize realloc cost during resize). UV sampling
    // is scaled by it so we never sample the transparent over-allocated
    // region, which would otherwise show as a squish + transparent strip.
    uv_max_and_scale: vec4<f32>,
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
    out.clip_position = u.transform * vec4<f32>(local * u.uv_max_and_scale.z, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// Catmull-Rom cubic weight (B = 0, C = 1/2): an interpolating reconstruction
// kernel (it passes through the source texels, so it is exact at integer phase)
// with a mild high-frequency boost. At fractional sub-pixel phases it keeps
// moving edges sharper than hardware bilinear — and sharper than the smoother
// Mitchell (B = C = 1/3) we used before, whose approximating center weight
// softened text during a translate. The negative lobes can ring slightly on
// mid-tone edges; on high-contrast (near black-on-white) text the overshoot is
// clamped away.
fn cubic_weight(x_in: f32) -> f32 {
    let b = 0.0;
    let c = 0.5;
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
    // Effective content dimensions inside the (possibly over-allocated)
    // texture. `uv_max == 1` recovers the pre-quantization sampling where
    // the content fills the whole texture.
    let content_dims = dims * u.uv_max_and_scale.xy;
    // Continuous texel coordinate (texel centers at integers), scaled so
    // quad-UV [0,1] traverses exactly the content sub-region rather than
    // the full texture.
    let coord = in.uv * content_dims - vec2<f32>(0.5);

    // When the cache lands exactly on the device-pixel grid (e.g. the widget
    // snapped a static transform), one texel maps to one device pixel: sample it
    // directly. This is a pure optimization — Catmull-Rom is already exact at
    // integer phase, so the 16-tap path below returns the same texel — it just
    // skips the loop for the common snapped/at-rest case.
    let nearest = round(coord);
    let frac = coord - nearest;
    if (max(abs(frac.x), abs(frac.y)) < 0.01) {
        let suv = (nearest + vec2<f32>(0.5)) / dims;
        return textureSampleLevel(u_texture, u_sampler, suv, 0.0);
    }

    // Otherwise (mid-animation, fractional offset) reconstruct with a 4x4
    // Catmull-Rom bicubic: sharp resampling of the moving edges.
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
