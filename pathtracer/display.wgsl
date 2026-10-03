// Averages the accumulated samples, applies exposure and the AgX tone curve,
// and outlines the selected material.

struct View {
  width: u32, height: u32, canvasW: u32, canvasH: u32,
  samples: f32, exposure: f32, mode: u32, selected: u32,
  accent: vec3f, pad: f32,
  lookPower: f32, lookSaturation: f32, pad1: f32, pad2: f32,
}

@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var<storage, read> accum: array<vec4f>;

const NONE = 0xffffffffu;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn texel(p: vec2i) -> vec4f {
  let q = clamp(p, vec2i(0), vec2i(i32(V.width) - 1, i32(V.height) - 1));
  return accum[u32(q.y) * V.width + u32(q.x)];
}

// AgX base look, polynomial fit by Benjamin Wrensch (iolite)
fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// A gentler version of AgX's "punchy" look: a little more contrast and saturation.
fn agxLook(v: vec3f) -> vec3f {
  let luma = dot(v, vec3f(0.2126, 0.7152, 0.0722));
  let p = pow(max(v, vec3f(0.0)), vec3f(V.lookPower));
  return luma + V.lookSaturation * (p - luma);
}

fn agx(c: vec3f) -> vec3f {
  let inset = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                      0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let outset = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                       -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                       -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393;
  let maxEv = 4.026069;
  var v = inset * max(c, vec3f(1e-10));
  v = (clamp(log2(v), vec3f(minEv), vec3f(maxEv)) - minEv) / (maxEv - minEv);
  return clamp(outset * agxLook(agxContrast(v)), vec3f(0.0), vec3f(1.0));
}

fn isSelected(p: vec2i) -> bool {
  return bitcast<u32>(texel(p).w) == V.selected;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let rp = pos.xy * vec2f(f32(V.width), f32(V.height)) / vec2f(f32(V.canvasW), f32(V.canvasH)) - 0.5;
  let i0 = vec2i(floor(rp));
  let f = rp - floor(rp);
  let top = mix(texel(i0).xyz, texel(i0 + vec2i(1, 0)).xyz, f.x);
  let bottom = mix(texel(i0 + vec2i(0, 1)).xyz, texel(i0 + vec2i(1, 1)).xyz, f.x);
  var c = mix(top, bottom, f.y) / max(V.samples, 1.0);
  if (V.mode == 0u) { c = agx(c * V.exposure); }    // the X-ray view is already display-ready
  if (V.selected != NONE) {
    let p = vec2i(round(rp));
    let me = isSelected(p);
    let edge = me != isSelected(p + vec2i(1, 0)) || me != isSelected(p - vec2i(1, 0))
            || me != isSelected(p + vec2i(0, 1)) || me != isSelected(p - vec2i(0, 1));
    if (edge) { c = mix(c, V.accent, 0.85); }
  }
  return vec4f(c, 1.0);
}
