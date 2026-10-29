// Edge-aware à-trous filter for the first few samples after the view changes
// (Dammertz et al. 2010, "Edge-avoiding à-trous wavelet transform", with SVGF's brightness test).
// Lighting is divided by each pixel's color first so the blur leaves textures alone, and
// neighbors only count if they show the same material on the same surface, unless both are out
// of focus, where the pixel is already a blend of many surfaces. main.js fades it out by 256
// samples, so the settled image is the plain average.

struct Params { width: u32, height: u32, step: u32, samples: f32 }
struct Pixel { sum: vec4f, guide: vec4f }    // see trace.wgsl

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> accum: array<Pixel>;
@group(0) @binding(2) var<storage, read> src: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4f>;

fn albedo(g: vec4f) -> vec4f { return unpack4x8unorm(bitcast<u32>(g.z)); }

fn octDecode(packed: u32) -> vec3f {
  let e = unpack2x16snorm(packed);
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn demodulated(i: u32) -> vec3f {
  let c = accum[i].sum.xyz / max(P.samples, 1.0);
  let g = accum[i].guide;
  if (g.y < 0.0) { return c; }
  return c / max(albedo(g).xyz, vec3f(0.01));
}

// Accumulated average divided by the pixel's color, plus how noisy its brightness is, estimated
// from the 5x5 neighbors on the same material (there is no history to measure it over time).
// A pixel far brighter than all its neighbors is one lucky path: it is pulled down to their range,
// or the edge-stopping below would keep it as a sparkle.
@compute @workgroup_size(8, 8)
fn demodulate(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let i = gid.y * P.width + gid.x;
  let mc = bitcast<u32>(accum[i].sum.w);
  var m1 = 0.0;
  var m2 = 0.0;
  var n = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      if (x == 0 && y == 0) { continue; }
      let q = vec2i(gid.xy) + vec2i(x, y);
      if (q.x < 0 || q.y < 0 || q.x >= i32(P.width) || q.y >= i32(P.height)) { continue; }
      let j = u32(q.y) * P.width + u32(q.x);
      if (bitcast<u32>(accum[j].sum.w) != mc) { continue; }
      let l = luminance(demodulated(j));
      m1 += l;
      m2 += l * l;
      n += 1.0;
    }
  }
  var d = demodulated(i);
  let l = luminance(d);
  if (n < 1.0) { dst[i] = vec4f(d, l * l); return; }
  let mean = m1 / n;
  let variance = max(m2 / n - mean * mean, 0.0);
  let cap = mean + 3.0 * sqrt(variance) + 0.05 * mean;
  if (l > cap) { d *= cap / l; }
  dst[i] = vec4f(d, variance);
}

// One à-trous pass. Brightness differences only stop the blur when they are larger than the
// noise (as in SVGF), so shadow edges survive while noise is averaged away.
@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let i = gid.y * P.width + gid.x;
  let gc = accum[i].guide;
  let cc = src[i];
  if (gc.y < 0.0) { dst[i] = cc; return; }    // sky: leave it alone
  let nc = octDecode(bitcast<u32>(gc.x));
  let mc = bitcast<u32>(accum[i].sum.w);
  let blur = albedo(gc).w;
  if (blur <= 0.0) { dst[i] = cc; return; }
  let lc = luminance(cc.xyz);
  let sigmaL = 4.0 * sqrt(cc.w) + 1e-4;
  let kernel = array<f32, 5>(1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0);
  var sum = vec4f(0.0);
  var wsum = 0.0;
  for (var y = -2; y <= 2; y++) {
    for (var x = -2; x <= 2; x++) {
      let q = vec2i(gid.xy) + vec2i(x, y) * i32(P.step);
      if (q.x < 0 || q.y < 0 || q.x >= i32(P.width) || q.y >= i32(P.height)) { continue; }
      let j = u32(q.y) * P.width + u32(q.x);
      let gq = accum[j].guide;
      if (gq.y < 0.0) { continue; }
      let cq = src[j];
      var w = kernel[x + 2] * kernel[y + 2];
      if (x != 0 || y != 0) {
        let nq = octDecode(bitcast<u32>(gq.x));
        let dist = length(vec2f(f32(x), f32(y))) * f32(P.step);
        let defocus = clamp(min(gc.w, gq.w) / f32(P.step) - 1.0, 0.0, 1.0);
        let same = select(defocus, 1.0, bitcast<u32>(accum[j].sum.w) == mc);
        w *= same * mix(pow(max(dot(nc, nq), 0.0), 8.0), 1.0, defocus);
        w *= exp(-abs(gc.y - gq.y) / (gc.y * (0.004 * dist + 0.2 * defocus) + 1e-4));
        w *= exp(-abs(lc - luminance(cq.xyz)) / sigmaL);
        w *= blur;
      }
      sum += vec4f(cq.xyz * w, cq.w * w * w);
      wsum += w;
    }
  }
  dst[i] = vec4f(sum.xyz / max(wsum, 1e-8), sum.w / max(wsum * wsum, 1e-8));
}
