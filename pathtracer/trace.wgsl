// Megakernel path tracer: one thread per pixel, a few samples per frame, accumulated over time.
// Geometry lives in bottom-level BVHs (one per mesh or sphere set) that instances place in the world.

struct Frame {
  camPos: vec3f, lensRadius: f32,
  camRight: vec3f, focusDist: f32,
  camUp: vec3f, tanHalfFov: f32,
  camFwd: vec3f, aspect: f32,
  width: u32, height: u32, frame: u32, spp: u32,
  numInstances: u32, numLights: u32, maxBounces: u32, mode: u32,
  envGridW: u32, envGridH: u32, envIntensity: f32, envRotation: f32,
  lightArea: f32, envVisible: f32, clampMax: f32, pickX: f32,
  bgColor: vec3f, pickY: f32,
  accumulate: u32, pad0: u32, pad1: u32, pad2: u32,
  groundCenter: vec3f, groundOn: f32,         // ground projection of the environment photo
  groundIrradiance: vec3f, pad3: f32,
}

struct Instance {
  r0: vec4f, r1: vec4f, r2: vec4f,     // world -> object affine rows
  bmin: vec3f, root: u32,
  bmax: vec3f, kind: u32,
  matBase: u32, flags: u32, pad0: u32, pad1: u32,
}

struct Material {
  color: vec3f, roughness: f32,
  emission: vec3f, metallic: f32,
  transmission: f32, ior: f32, density: f32, scatter: f32,
  anisotropy: f32, projected: f32, pad1: f32, pad2: f32,
}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var<storage, read> nodes: array<vec4f>;
@group(0) @binding(2) var<storage, read> prims: array<vec4f>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read> materials: array<Material>;
@group(0) @binding(5) var<storage, read> lights: array<vec4f>;
@group(0) @binding(6) var<storage, read> envCdf: array<f32>;
@group(0) @binding(7) var envTex: texture_2d<f32>;
@group(0) @binding(8) var envSamp: sampler;
@group(0) @binding(9) var<storage, read_write> accum: array<vec4f>;
@group(0) @binding(10) var<storage, read_write> stats: array<atomic<u32>, 8>;

const PI = 3.14159265359;
const BIG = 1e20;
const NONE = 0xffffffffu;
const KIND_SPHERES = 1u;
const FLAG_LIGHT = 1u;
const STACK = 40u;

// ---------------------------------------------------------------- random numbers

var<private> rng: u32;
var<private> visits: u32;
var<private> rays: u32;
var<private> primaryMat: u32;

fn hash(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rand() -> f32 {
  rng = rng * 747796405u + 2891336453u;
  let w = ((rng >> ((rng >> 28u) + 4u)) ^ rng) * 277803737u;
  return f32(((w >> 22u) ^ w) >> 8u) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------- intersection

struct Hit { t: f32, u: f32, v: f32, prim: u32, inst: u32 }

fn intersectTriangle(p: u32, o: vec3f, d: vec3f, hit: ptr<function, Hit>, inst: u32) -> bool {
  let v0 = prims[4u * p].xyz;
  let e1 = prims[4u * p + 1u].xyz;
  let e2 = prims[4u * p + 2u].xyz;
  let pv = cross(d, e2);
  let det = dot(e1, pv);
  if (det == 0.0) { return false; }
  let inv = 1.0 / det;
  let tv = o - v0;
  let u = dot(tv, pv) * inv;
  if (u < 0.0 || u > 1.0) { return false; }
  let qv = cross(tv, e1);
  let v = dot(d, qv) * inv;
  if (v < 0.0 || u + v > 1.0) { return false; }
  let t = dot(e2, qv) * inv;
  if (t <= 0.0 || t >= (*hit).t) { return false; }
  *hit = Hit(t, u, v, p, inst);
  return true;
}

fn intersectSphere(p: u32, o: vec3f, d: vec3f, hit: ptr<function, Hit>, inst: u32) -> bool {
  let s = prims[4u * p];
  let f = o - s.xyz;
  let a = dot(d, d);
  let tc = -dot(f, d) / a;
  let l = f + tc * d;                       // closest approach, computed directly for precision
  let h2 = s.w * s.w - dot(l, l);
  if (h2 < 0.0) { return false; }
  let q = sqrt(h2 / a);
  var t = tc - q;
  if (t <= 0.0) { t = tc + q; }
  if (t <= 0.0 || t >= (*hit).t) { return false; }
  *hit = Hit(t, 0.0, 0.0, p, inst);
  return true;
}

fn safeInverse(d: vec3f) -> vec3f {
  let tiny = select(vec3f(-1e-12), vec3f(1e-12), d >= vec3f(0.0));
  return 1.0 / select(d, tiny, abs(d) < vec3f(1e-12));
}

// Walk one bottom-level BVH. Both children's boxes come in one 64-byte node;
// leaves are tested on the spot and the nearer inner child is visited first.
fn traverse(o: vec3f, d: vec3f, root: u32, kind: u32, inst: u32, hit: ptr<function, Hit>, anyHit: bool) -> bool {
  let invd = safeInverse(d);
  let oid = o * invd;
  var stack: array<u32, STACK>;
  var sp = 0u;
  var node = root;
  var found = false;
  for (var guard = 0u; guard < 2048u; guard++) {   // backstop: real traversals need < 200 steps
    visits += 1u;
    let lo0 = nodes[4u * node];
    let hi0 = nodes[4u * node + 1u];
    let lo1 = nodes[4u * node + 2u];
    let hi1 = nodes[4u * node + 3u];
    let la = lo0.xyz * invd - oid; let lb = hi0.xyz * invd - oid;
    let ra = lo1.xyz * invd - oid; let rb = hi1.xyz * invd - oid;
    let lmn = min(la, lb); let lmx = max(la, lb);
    let rmn = min(ra, rb); let rmx = max(ra, rb);
    let lNear = max(max(lmn.x, lmn.y), max(lmn.z, 0.0));
    let lFar = min(min(lmx.x, lmx.y), min(lmx.z, (*hit).t));
    let rNear = max(max(rmn.x, rmn.y), max(rmn.z, 0.0));
    let rFar = min(min(rmx.x, rmx.y), min(rmx.z, (*hit).t));
    var goL = lNear <= lFar;
    var goR = rNear <= rFar;
    let lCount = bitcast<u32>(hi0.w);
    let rCount = bitcast<u32>(hi1.w);
    if (goL && lCount > 0u) {
      let first = bitcast<u32>(lo0.w);
      for (var i = first; i < first + lCount; i++) {
        visits += 1u;
        if (kind == KIND_SPHERES) { found = intersectSphere(i, o, d, hit, inst) || found; }
        else { found = intersectTriangle(i, o, d, hit, inst) || found; }
      }
      if (found && anyHit) { return true; }
      goL = false;
    }
    if (goR && rCount > 0u) {
      let first = bitcast<u32>(lo1.w);
      for (var i = first; i < first + rCount; i++) {
        visits += 1u;
        if (kind == KIND_SPHERES) { found = intersectSphere(i, o, d, hit, inst) || found; }
        else { found = intersectTriangle(i, o, d, hit, inst) || found; }
      }
      if (found && anyHit) { return true; }
      goR = false;
    }
    if (goL && goR) {
      var near = bitcast<u32>(lo0.w);
      var far = bitcast<u32>(lo1.w);
      if (rNear < lNear) { let t = near; near = far; far = t; }
      if (sp < STACK) { stack[sp] = far; sp += 1u; }
      node = near;
    } else if (goL) {
      node = bitcast<u32>(lo0.w);
    } else if (goR) {
      node = bitcast<u32>(lo1.w);
    } else {
      if (sp == 0u) { break; }
      sp -= 1u;
      node = stack[sp];
    }
  }
  return found;
}

fn toObject(I: Instance, p: vec3f) -> vec3f {
  return vec3f(dot(I.r0.xyz, p) + I.r0.w, dot(I.r1.xyz, p) + I.r1.w, dot(I.r2.xyz, p) + I.r2.w);
}

fn dirToObject(I: Instance, d: vec3f) -> vec3f {
  return vec3f(dot(I.r0.xyz, d), dot(I.r1.xyz, d), dot(I.r2.xyz, d));
}

// Top level: few instances, so a linear walk over their world boxes.
fn intersectScene(ro: vec3f, rd: vec3f, tmax: f32, anyHit: bool) -> Hit {
  var hit = Hit(tmax, 0.0, 0.0, 0u, NONE);
  let invd = safeInverse(rd);
  rays += 1u;
  for (var i = 0u; i < F.numInstances; i++) {
    let I = instances[i];
    let ta = (I.bmin - ro) * invd;
    let tb = (I.bmax - ro) * invd;
    let mn = min(ta, tb); let mx = max(ta, tb);
    let near = max(max(mn.x, mn.y), max(mn.z, 0.0));
    let far = min(min(mx.x, mx.y), min(mx.z, hit.t));
    if (near > far) { continue; }
    if (traverse(toObject(I, ro), dirToObject(I, rd), I.root, I.kind, i, &hit, anyHit) && anyHit) { return hit; }
  }
  return hit;
}

// ---------------------------------------------------------------- surfaces

struct Surface { p: vec3f, ng: vec3f, ns: vec3f, mat: u32 }

fn octDecode(packed: u32) -> vec3f {
  let e = unpack2x16snorm(packed);
  var n = vec3f(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

fn normalToWorld(I: Instance, n: vec3f) -> vec3f {
  return normalize(I.r0.xyz * n.x + I.r1.xyz * n.y + I.r2.xyz * n.z);   // inverse transpose of object->world
}

fn surfaceAt(hit: Hit, ro: vec3f, rd: vec3f) -> Surface {
  let I = instances[hit.inst];
  var s: Surface;
  s.p = ro + rd * hit.t;
  let base = 4u * hit.prim;
  if (I.kind == KIND_SPHERES) {
    let sph = prims[base];
    let n = (toObject(I, s.p) - sph.xyz) / sph.w;
    s.ng = normalToWorld(I, n);
    s.ns = s.ng;
    s.mat = I.matBase + bitcast<u32>(prims[base + 1u].x);
  } else {
    let v0 = prims[base];
    let e1 = prims[base + 1u].xyz;
    let e2 = prims[base + 2u].xyz;
    let nn = bitcast<vec4u>(prims[base + 3u]);
    s.ng = normalToWorld(I, cross(e1, e2));
    let w = 1.0 - hit.u - hit.v;
    s.ns = normalToWorld(I, octDecode(nn.x) * w + octDecode(nn.y) * hit.u + octDecode(nn.z) * hit.v);
    s.mat = I.matBase + bitcast<u32>(v0.w);
  }
  return s;
}

// Wächter & Binder, "A Fast and Robust Method for Avoiding Self-Intersection" (Ray Tracing Gems)
fn offsetRay(p: vec3f, n: vec3f) -> vec3f {
  let off = vec3i(256.0 * n);
  let pi = vec3f(
    bitcast<f32>(bitcast<i32>(p.x) + select(off.x, -off.x, p.x < 0.0)),
    bitcast<f32>(bitcast<i32>(p.y) + select(off.y, -off.y, p.y < 0.0)),
    bitcast<f32>(bitcast<i32>(p.z) + select(off.z, -off.z, p.z < 0.0)));
  return select(pi, p + (1.0 / 65536.0) * n, abs(p) < vec3f(1.0 / 32.0));
}

// ---------------------------------------------------------------- sampling helpers

struct Basis { t: vec3f, b: vec3f, n: vec3f }

fn basis(n: vec3f) -> Basis {        // Duff et al. 2017
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let a = -1.0 / (s + n.z);
  let b = n.x * n.y * a;
  return Basis(vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x), vec3f(b, s + n.y * n.y * a, -n.y), n);
}

fn toLocal(B: Basis, v: vec3f) -> vec3f { return vec3f(dot(v, B.t), dot(v, B.b), dot(v, B.n)); }
fn toWorld(B: Basis, v: vec3f) -> vec3f { return B.t * v.x + B.b * v.y + B.n * v.z; }

fn cosineHemisphere() -> vec3f {
  let r = sqrt(rand());
  let phi = 2.0 * PI * rand();
  return vec3f(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - r * r)));
}

fn uniformSphere() -> vec3f {
  let z = 1.0 - 2.0 * rand();
  let r = sqrt(max(0.0, 1.0 - z * z));
  let phi = 2.0 * PI * rand();
  return vec3f(r * cos(phi), r * sin(phi), z);
}

fn luminance(c: vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

fn powerHeuristic(a: f32, b: f32) -> f32 {
  let a2 = a * a;
  return a2 / max(a2 + b * b, 1e-20);
}

// ---------------------------------------------------------------- GGX microfacets

fn ggxD(nh: f32, a2: f32) -> f32 {
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn smithG1(nv: f32, a2: f32) -> f32 {
  return 2.0 * nv / (nv + sqrt(a2 + (1.0 - a2) * nv * nv));
}

// Visible-normal sampling with spherical caps (Dupuy & Benyoub 2023). wo in local space, z up.
fn sampleVNDF(wo: vec3f, alpha: f32) -> vec3f {
  let wi = normalize(vec3f(wo.xy * alpha, wo.z));
  let phi = 2.0 * PI * rand();
  let z = (1.0 - rand()) * (1.0 + wi.z) - wi.z;
  let st = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let h = vec3f(st * cos(phi), st * sin(phi), z) + wi;
  return normalize(vec3f(h.xy * alpha, max(h.z, 1e-6)));
}

fn schlick(f0: vec3f, c: f32) -> vec3f {
  let m = clamp(1.0 - c, 0.0, 1.0);
  let m2 = m * m;
  return f0 + (1.0 - f0) * (m2 * m2 * m);
}

fn fresnelDielectric(cosI: f32, eta: f32) -> f32 {      // eta = n_incident / n_transmitted
  let sin2T = eta * eta * (1.0 - cosI * cosI);
  if (sin2T >= 1.0) { return 1.0; }
  let cosT = sqrt(1.0 - sin2T);
  let rs = (eta * cosI - cosT) / (eta * cosI + cosT);
  let rp = (cosI - eta * cosT) / (cosI + eta * cosT);
  return 0.5 * (rs * rs + rp * rp);
}

// ---------------------------------------------------------------- opaque BSDF: Lambert + GGX

struct BsdfEval { f: vec3f, pdf: f32 }    // f already includes the cosine

fn isMatte(M: Material) -> bool { return M.roughness >= 0.999 && M.metallic <= 0.001; }

fn specularProbability(M: Material, nv: f32) -> f32 {
  if (M.metallic >= 0.999) { return 1.0; }
  if (isMatte(M)) { return 0.0; }
  let f0 = mix(vec3f(0.04), M.color, M.metallic);
  let spec = luminance(schlick(f0, nv));
  let diff = (1.0 - M.metallic) * luminance(M.color) * (1.0 - spec);
  return clamp(spec / max(spec + diff, 1e-6), 0.1, 0.9);
}

fn evalOpaque(M: Material, wo: vec3f, wi: vec3f) -> BsdfEval {
  if (wi.z <= 0.0 || wo.z <= 0.0) { return BsdfEval(vec3f(0.0), 0.0); }
  if (isMatte(M)) { return BsdfEval(M.color / PI * wi.z, wi.z / PI); }
  let alpha = max(M.roughness * M.roughness, 1e-3);
  let a2 = alpha * alpha;
  let h = normalize(wo + wi);
  let f0 = mix(vec3f(0.04), M.color, M.metallic);
  let Fr = schlick(f0, dot(wo, h));
  let D = ggxD(h.z, a2);
  let g1o = smithG1(wo.z, a2);
  let spec = Fr * D * g1o * smithG1(wi.z, a2) / (4.0 * wo.z * wi.z);
  let diff = (1.0 - M.metallic) * (1.0 - Fr) * M.color / PI;
  let ps = specularProbability(M, wo.z);
  let pdf = ps * g1o * D / (4.0 * wo.z) + (1.0 - ps) * wi.z / PI;
  return BsdfEval((spec + diff) * wi.z, pdf);
}

fn sampleOpaque(M: Material, wo: vec3f) -> vec3f {
  if (rand() < specularProbability(M, wo.z)) {
    let alpha = max(M.roughness * M.roughness, 1e-3);
    let h = sampleVNDF(wo, alpha);
    return reflect(-wo, h);
  }
  return cosineHemisphere();
}

// ---------------------------------------------------------------- environment

fn envUV(d: vec3f) -> vec2f {
  let u = fract(atan2(d.x, -d.z) / (2.0 * PI) + 0.5 - F.envRotation);
  let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
  return vec2f(u, v);
}

fn envRadiance(d: vec3f) -> vec3f {
  return textureSampleLevel(envTex, envSamp, envUV(d), 0.0).rgb * F.envIntensity;
}

// The floor under the photo's camera, seen from where the photo was taken, is the photo's own
// ground. Giving the floor that colour as its albedo (scaled by the sky's irradiance) makes it
// match the photo wherever it is lit, while the objects still cast real shadows onto it.
fn groundAlbedo(p: vec3f) -> vec3f {
  let d = normalize(p - F.groundCenter);
  let photo = textureSampleLevel(envTex, envSamp, envUV(d), 0.0).rgb;
  return min(PI * photo / max(F.groundIrradiance, vec3f(1e-6)), vec3f(0.98));
}

fn envCellPdf(row: u32, col: u32) -> f32 {
  let gw = F.envGridW;
  let base = F.envGridH + 1u + row * (gw + 1u);
  return (envCdf[row + 1u] - envCdf[row]) * (envCdf[base + col + 1u] - envCdf[base + col]);
}

fn envPdf(d: vec3f) -> f32 {
  let uv = envUV(d);
  let col = min(u32(uv.x * f32(F.envGridW)), F.envGridW - 1u);
  let row = min(u32(uv.y * f32(F.envGridH)), F.envGridH - 1u);
  let sinT = sqrt(max(1.0 - d.y * d.y, 1e-8));
  return envCellPdf(row, col) * f32(F.envGridW * F.envGridH) / (2.0 * PI * PI * sinT);
}

fn searchCdf(base: u32, n: u32, x: f32) -> u32 {
  var lo = 0u;
  var hi = n;
  while (hi - lo > 1u) {
    let mid = (lo + hi) / 2u;
    if (envCdf[base + mid] <= x) { lo = mid; } else { hi = mid; }
  }
  return lo;
}

struct LightSample { wi: vec3f, Li: vec3f, pdf: f32, dist: f32 }

fn sampleEnv() -> LightSample {
  let gw = F.envGridW; let gh = F.envGridH;
  let row = searchCdf(0u, gh, rand());
  let col = searchCdf(gh + 1u + row * (gw + 1u), gw, rand());
  let u = (f32(col) + rand()) / f32(gw);
  let v = (f32(row) + rand()) / f32(gh);
  let phi = (u - 0.5 + F.envRotation) * 2.0 * PI;
  let theta = v * PI;
  let st = sin(theta);
  let d = vec3f(st * sin(phi), cos(theta), -st * cos(phi));
  let pdf = envCellPdf(row, col) * f32(gw * gh) / (2.0 * PI * PI * max(st, 1e-4));
  return LightSample(d, envRadiance(d), pdf, BIG);
}

fn envSelectProbability() -> f32 {
  if (F.envIntensity <= 0.0) { return 0.0; }
  return select(1.0, 0.5, F.numLights > 0u);
}

// Emissive triangles are picked in proportion to area, so every point on every light is equally likely.
fn sampleLight(p: vec3f) -> LightSample {
  let pEnv = envSelectProbability();
  if (rand() < pEnv) {
    var s = sampleEnv();
    s.pdf *= pEnv;
    return s;
  }
  if (F.numLights == 0u) { return LightSample(vec3f(0.0, 1.0, 0.0), vec3f(0.0), 0.0, 0.0); }
  let x = rand();
  var lo = 0u;
  var hi = F.numLights - 1u;
  while (lo < hi) {                      // first light whose cumulative area exceeds x
    let mid = (lo + hi) / 2u;
    if (lights[4u * mid + 1u].w <= x) { lo = mid + 1u; } else { hi = mid; }
  }
  let L0 = lights[4u * lo];
  let e1 = lights[4u * lo + 1u].xyz;
  let e2 = lights[4u * lo + 2u].xyz;
  let n = lights[4u * lo + 3u].xyz;
  var a = rand(); var b = rand();
  if (a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
  let q = L0.xyz + e1 * a + e2 * b;
  let dv = q - p;
  let dist = length(dv);
  let wi = dv / dist;
  let cosL = dot(-wi, n);
  if (cosL <= 0.0) { return LightSample(wi, vec3f(0.0), 0.0, dist); }
  let Le = materials[bitcast<u32>(L0.w)].emission;
  return LightSample(wi, Le, (1.0 - pEnv) * dist * dist / (cosL * F.lightArea), dist);
}

// ---------------------------------------------------------------- path tracing

fn addLight(L: ptr<function, vec3f>, c: vec3f, depth: u32) {
  var v = c;
  if (depth > 0u) {                      // tame fireflies from rare caustic paths
    let l = luminance(v);
    if (l > F.clampMax) { v *= F.clampMax / l; }
  }
  *L += v;
}

fn radiance(ro0: vec3f, rd0: vec3f) -> vec3f {
  var ro = ro0;
  var rd = rd0;
  var tp = vec3f(1.0);
  var L = vec3f(0.0);
  var specular = true;        // the last bounce could not be light-sampled
  var prevPdf = 0.0;
  var medium = NONE;
  var depth = 0u;             // surfaces hit so far
  var bounces = 0u;           // diffuse or glossy bounces; mirror-like glass doesn't count
  var events = 0u;
  let pEnv = envSelectProbability();

  loop {
    events += 1u;
    if (events > 256u) { break; }
    let hit = intersectScene(ro, rd, BIG, false);

    // inside glass: absorb or scatter along the way
    if (medium != NONE) {
      if (hit.inst == NONE) {
        medium = NONE;           // escaped through a hole in an open mesh
      } else {
        let M = materials[medium];
        if (M.scatter > 0.0) {
          let s = -log(1.0 - rand()) / max(M.density, 1e-4);
          if (s < hit.t) {
            ro = ro + rd * s;
            tp *= M.color;
            let g = M.anisotropy;
            var dir = uniformSphere();
            if (abs(g) > 1e-3) {     // Henyey-Greenstein
              let sq = (1.0 - g * g) / (1.0 - g + 2.0 * g * rand());
              let cosT = (1.0 + g * g - sq * sq) / (2.0 * g);
              let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
              let phi = 2.0 * PI * rand();
              dir = toWorld(basis(rd), vec3f(sinT * cos(phi), sinT * sin(phi), cosT));
            }
            rd = dir;
            specular = true;
            let m = max(tp.x, max(tp.y, tp.z));
            if (m < 0.1) {             // gamble only once the walk has lost most of its light
              if (rand() * 0.1 > m) { break; }
              tp *= 0.1 / m;
            }
            continue;
          }
        } else if (M.density > 0.0) {
          tp *= exp(log(max(M.color, vec3f(1e-3))) * M.density * hit.t);
        }
      }
    }

    if (hit.inst == NONE) {
      var Le = envRadiance(rd);
      if (depth == 0u && F.envVisible < 0.5) { Le = F.bgColor; }
      var w = 1.0;
      if (!specular && pEnv > 0.0) { w = powerHeuristic(prevPdf, envPdf(rd) * pEnv); }
      addLight(&L, tp * Le * w, depth);
      break;
    }

    let s = surfaceAt(hit, ro, rd);
    if (events == 1u) { primaryMat = s.mat; }
    var M = materials[s.mat];
    if (M.projected > 0.5 && F.groundOn > 0.5) { M.color = groundAlbedo(s.p); }
    let wo = -rd;
    let front = dot(s.ng, wo) > 0.0;
    let I = instances[hit.inst];

    if (front && any(M.emission > vec3f(0.0))) {
      var w = 1.0;
      if (!specular && (I.flags & FLAG_LIGHT) != 0u) {
        let cosL = dot(s.ng, wo);
        let lightPdf = (1.0 - pEnv) * hit.t * hit.t / (cosL * F.lightArea);
        w = powerHeuristic(prevPdf, lightPdf);
      }
      addLight(&L, tp * M.emission * w, depth);
    }

    if (bounces >= F.maxBounces) { break; }
    depth += 1u;

    let ng = select(-s.ng, s.ng, front);       // both normals on the side the ray came from
    var ns = select(-s.ns, s.ns, front);
    if (dot(ns, wo) <= 1e-4) { ns = ng; }

    if (M.transmission > 0.5 && M.scatter > 0.0 && !front) {
      // Leaving a scattering medium: treat the exit as diffuse transmission, as random-walk
      // subsurface scattering does, so the exit point can sample lights directly.
      let out = -ng;
      let nOut = -ns;
      let p = offsetRay(s.p, out);
      let ls = sampleLight(s.p);
      let cosL = dot(ls.wi, nOut);
      if (ls.pdf > 0.0 && cosL > 0.0 && dot(ls.wi, out) > 0.0) {
        let shadow = intersectScene(p, ls.wi, ls.dist * 0.9999, true);
        if (shadow.inst == NONE) {
          let bsdfPdf = cosL / PI;
          addLight(&L, tp * (cosL / PI) * ls.Li * powerHeuristic(ls.pdf, bsdfPdf) / ls.pdf, depth);
        }
      }
      bounces += 1u;
      let B = basis(nOut);
      let wi = toWorld(B, cosineHemisphere());
      if (dot(wi, out) <= 0.0) { break; }
      prevPdf = dot(wi, nOut) / PI;
      specular = false;
      medium = NONE;
      ro = p;
      rd = wi;
    } else if (M.transmission > 0.5) {
      // smooth or rough dielectric: reflect or refract through a sampled microfacet
      let B = basis(ns);
      let woL = toLocal(B, wo);
      let alpha = M.roughness * M.roughness;
      var m = ns;
      if (alpha > 1e-3) { m = toWorld(B, sampleVNDF(woL, alpha)); }
      let eta = select(M.ior, 1.0 / M.ior, front);
      let cosI = dot(wo, m);
      let Fr = fresnelDielectric(cosI, eta);
      var wi: vec3f;
      if (rand() < Fr) {
        wi = reflect(-wo, m);
        if (dot(wi, ng) <= 0.0) { break; }
        ro = offsetRay(s.p, ng);
      } else {
        wi = refract(-wo, m, eta);
        if (dot(wi, ng) >= 0.0 || all(wi == vec3f(0.0))) { break; }
        ro = offsetRay(s.p, -ng);
        if (M.density <= 0.0) { tp *= M.color; }
        medium = select(NONE, s.mat, front);
      }
      if (alpha > 1e-3) { tp *= smithG1(abs(dot(wi, ns)), alpha * alpha); }
      rd = wi;
      specular = true;
    } else {
      bounces += 1u;
      let B = basis(ns);
      let woL = toLocal(B, wo);
      let p = offsetRay(s.p, ng);

      // next event estimation with multiple importance sampling
      let ls = sampleLight(s.p);
      if (ls.pdf > 0.0 && dot(ls.wi, ng) > 0.0) {
        let e = evalOpaque(M, woL, toLocal(B, ls.wi));
        if (e.pdf > 0.0) {
          let shadow = intersectScene(p, ls.wi, ls.dist * 0.9999, true);
          if (shadow.inst == NONE) {
            addLight(&L, tp * e.f * ls.Li * powerHeuristic(ls.pdf, e.pdf) / ls.pdf, depth);
          }
        }
      }

      let wiL = sampleOpaque(M, woL);
      let e = evalOpaque(M, woL, wiL);
      let wi = toWorld(B, wiL);
      if (e.pdf <= 0.0 || dot(wi, ng) <= 0.0) { break; }
      tp *= e.f / e.pdf;
      prevPdf = e.pdf;
      specular = false;
      ro = p;
      rd = wi;
    }

    if (bounces > 3u) {
      let q = min(max(tp.x, max(tp.y, tp.z)), 0.95);
      if (rand() > q) { break; }
      tp /= q;
    }
  }
  return L;
}

// ---------------------------------------------------------------- X-ray view

fn turbo(x0: f32) -> vec3f {
  let x = clamp(x0, 0.0, 1.0);
  let v4 = vec4f(1.0, x, x * x, x * x * x);
  let v2 = v4.zw * v4.z;
  return vec3f(
    dot(v4, vec4f(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2f(-152.94239396, 59.28637943)),
    dot(v4, vec4f(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2f(4.27729857, 2.82956604)),
    dot(v4, vec4f(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2f(-89.90310912, 27.34824973)));
}

// X-ray view: how much tree each camera ray had to walk, from blue (little) to red (a lot).
fn xray(ro: vec3f, rd: vec3f) -> vec3f {
  visits = 0u;
  let hit = intersectScene(ro, rd, BIG, false);
  if (hit.inst != NONE) { primaryMat = surfaceAt(hit, ro, rd).mat; }
  return turbo(f32(visits) / 160.0);
}

// ---------------------------------------------------------------- entry points

struct Ray { o: vec3f, d: vec3f }

fn cameraRay(px: vec2f, lens: bool) -> Ray {
  let ndc = vec2f(px.x / f32(F.width) * 2.0 - 1.0, 1.0 - px.y / f32(F.height) * 2.0);
  let d = normalize(F.camFwd + F.camRight * (ndc.x * F.tanHalfFov * F.aspect) + F.camUp * (ndc.y * F.tanHalfFov));
  if (!lens || F.lensRadius <= 0.0) { return Ray(F.camPos, d); }
  let focus = F.camPos + d * (F.focusDist / dot(d, F.camFwd));
  let r = sqrt(rand()) * F.lensRadius;
  let phi = 2.0 * PI * rand();
  let o = F.camPos + F.camRight * (r * cos(phi)) + F.camUp * (r * sin(phi));
  return Ray(o, normalize(focus - o));
}

var<workgroup> groupRays: atomic<u32>;

@compute @workgroup_size(8, 4)
fn trace(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) li: u32) {
  let inside = gid.x < F.width && gid.y < F.height;
  if (inside) {
    let idx = gid.y * F.width + gid.x;
    rng = hash(idx ^ hash(F.frame * 0x9e3779b9u));
    var sum = vec3f(0.0);
    var id = NONE;
    for (var s = 0u; s < F.spp; s++) {
      // the first sample of a fresh image records which material the pixel shows,
      // for the selection outline
      let first = s == 0u && F.accumulate == 0u;
      let r = cameraRay(vec2f(gid.xy) + vec2f(rand(), rand()), true);
      primaryMat = NONE;
      var c: vec3f;
      if (F.mode == 0u) { c = radiance(r.o, r.d); } else { c = xray(r.o, r.d); }
      if (first) { id = primaryMat; }
      let bits = bitcast<vec3u>(c) & vec3u(0x7f800000u);
      if (!any(bits == vec3u(0x7f800000u))) { sum += c; }    // drop NaN and infinite samples
    }
    if (F.accumulate == 0u) {
      accum[idx] = vec4f(sum, bitcast<f32>(id));
    } else {
      let prev = accum[idx];
      accum[idx] = vec4f(prev.xyz + sum, prev.w);
    }
  }
  atomicAdd(&groupRays, rays);
  workgroupBarrier();
  if (li == 0u) { atomicAdd(&stats[0], atomicLoad(&groupRays)); }
}

@compute @workgroup_size(1)
fn pick() {
  let r = cameraRay(vec2f(F.pickX, F.pickY), false);
  let h = intersectScene(r.o, r.d, BIG, false);
  if (h.inst == NONE) {
    atomicStore(&stats[4], NONE);
    return;
  }
  atomicStore(&stats[4], h.inst);
  atomicStore(&stats[5], surfaceAt(h, r.o, r.d).mat);
  atomicStore(&stats[6], bitcast<u32>(h.t * dot(r.d, F.camFwd)));
}
