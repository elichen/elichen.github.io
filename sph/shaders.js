export const WORKGROUP_SIZE = 256;
export const MAX_OBSTACLES = 4;

const PARAMS_STRUCT = /* wgsl */ `
const PI = 3.14159265358979323846;
const MAX_OBSTACLES = ${MAX_OBSTACLES}u;

struct Particle {
  pos: vec2<f32>,
  vel: vec2<f32>,
};

struct Params {
  count: u32,
  gridW: u32,
  gridH: u32,
  obstacleCount: u32,

  boxSize: vec2<f32>,
  screenSize: vec2<f32>,

  h: f32,
  mass: f32,
  restDensity: f32,
  stiffness: f32,

  viscosity: f32,
  gravity: f32,
  tilt: f32,
  dt: f32,

  boundaryStiffness: f32,
  boundaryDamping: f32,
  maxSpeed: f32,
  damping: f32,

  mousePos: vec2<f32>,
  mouseVel: vec2<f32>,

  mouseRadius: f32,
  mouseStrength: f32,
  mouseMode: u32,
  pad0: u32,

  viewScale: f32,
  iso: f32,
  viewOffset: vec2<f32>,

  obstacles: array<vec4<f32>, ${MAX_OBSTACLES}>,
  obstacleVels: array<vec4<f32>, ${MAX_OBSTACLES}>,
};
`;

const GRID_HELPERS = /* wgsl */ `
fn cellIndexFor(pos: vec2<f32>) -> u32 {
  let cell = vec2<i32>(floor(pos / P.h));
  let cx = clamp(cell.x, 0, i32(P.gridW) - 1);
  let cy = clamp(cell.y, 0, i32(P.gridH) - 1);
  return u32(cy) * P.gridW + u32(cx);
}
`;

const SPH_HELPERS = /* wgsl */ `
fn poly6(r2: f32) -> f32 {
  let h2 = P.h * P.h;
  if (r2 > h2) {
    return 0.0;
  }
  let x = h2 - r2;
  return (4.0 / (PI * pow(P.h, 8.0))) * x * x * x;
}

fn spikyGrad(rvec: vec2<f32>, r: f32) -> vec2<f32> {
  let q = P.h - r;
  return -(30.0 / (PI * pow(P.h, 5.0))) * q * q * (rvec / r);
}

fn viscosityLap(r: f32) -> f32 {
  return (40.0 / (PI * pow(P.h, 5.0))) * (P.h - r);
}
`;

const RENDER_HELPERS = /* wgsl */ `
fn worldToPixel(pos: vec2<f32>) -> vec2<f32> {
  return P.viewOffset + vec2<f32>(pos.x, P.boxSize.y - pos.y) * P.viewScale;
}

fn pixelToClip(px: vec2<f32>) -> vec4<f32> {
  let ndc = vec2<f32>(
    (px.x / P.screenSize.x) * 2.0 - 1.0,
    1.0 - (px.y / P.screenSize.y) * 2.0
  );
  return vec4<f32>(ndc, 0.0, 1.0);
}

fn pixelToWorld(px: vec2<f32>) -> vec2<f32> {
  let local = (px - P.viewOffset) / P.viewScale;
  return vec2<f32>(local.x, P.boxSize.y - local.y);
}
`;

export const countWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read_write> cellCount: array<atomic<u32>>;

${GRID_HELPERS}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }

  let cell = cellIndexFor(particles[i].pos);
  _ = atomicAdd(&cellCount[cell], 1u);
}
`;

export const scanWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> cellCount: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellStart: array<u32>;

@compute @workgroup_size(1)
fn main() {
  let cells = P.gridW * P.gridH;
  var sum = 0u;
  for (var i = 0u; i < cells; i = i + 1u) {
    cellStart[i] = sum;
    sum = sum + cellCount[i];
  }
}
`;

export const scatterWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> cellOffset: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sortedIndices: array<u32>;

${GRID_HELPERS}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }

  let cell = cellIndexFor(particles[i].pos);
  let slot = cellStart[cell] + atomicAdd(&cellOffset[cell], 1u);
  if (slot < P.count) {
    sortedIndices[slot] = i;
  }
}
`;

export const densityWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> P: Params;
@group(0) @binding(2) var<storage, read> cellCount: array<u32>;
@group(0) @binding(3) var<storage, read> cellStart: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(5) var<storage, read_write> density: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressure: array<f32>;

${GRID_HELPERS}
${SPH_HELPERS}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }

  let pi = particles[i].pos;
  let base = vec2<i32>(floor(pi / P.h));
  var rho: f32 = 0.0;

  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    let cy = base.y + oy;
    if (cy < 0 || cy >= i32(P.gridH)) {
      continue;
    }

    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let cx = base.x + ox;
      if (cx < 0 || cx >= i32(P.gridW)) {
        continue;
      }

      let cell = u32(cy) * P.gridW + u32(cx);
      let start = cellStart[cell];
      let end = start + cellCount[cell];
      for (var s = start; s < end; s = s + 1u) {
        let j = sortedIndices[s];
        let rvec = pi - particles[j].pos;
        rho = rho + P.mass * poly6(dot(rvec, rvec));
      }
    }
  }

  let safeRho = max(rho, P.restDensity * 0.1);
  density[i] = safeRho;
  pressure[i] = max(P.stiffness * (safeRho - P.restDensity), 0.0);
}
`;

export const integrateWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(1) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(2) var<uniform> P: Params;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> cellStart: array<u32>;
@group(0) @binding(5) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(6) var<storage, read> density: array<f32>;
@group(0) @binding(7) var<storage, read> pressure: array<f32>;

${GRID_HELPERS}
${SPH_HELPERS}

fn wallAcceleration(pos: vec2<f32>, vel: vec2<f32>) -> vec2<f32> {
  var acc = vec2<f32>(0.0);

  if (pos.x < P.h) {
    acc += P.boundaryStiffness * (P.h - pos.x) * vec2<f32>(1.0, 0.0) - P.boundaryDamping * vel;
  }
  if (pos.x > P.boxSize.x - P.h) {
    acc += P.boundaryStiffness * (pos.x - (P.boxSize.x - P.h)) * vec2<f32>(-1.0, 0.0) - P.boundaryDamping * vel;
  }
  if (pos.y < P.h) {
    acc += P.boundaryStiffness * (P.h - pos.y) * vec2<f32>(0.0, 1.0) - P.boundaryDamping * vel;
  }
  if (pos.y > P.boxSize.y - P.h) {
    acc += P.boundaryStiffness * (pos.y - (P.boxSize.y - P.h)) * vec2<f32>(0.0, -1.0) - P.boundaryDamping * vel;
  }

  return acc;
}

fn obstacleAcceleration(pos: vec2<f32>, vel: vec2<f32>) -> vec2<f32> {
  var acc = vec2<f32>(0.0);

  for (var oi = 0u; oi < MAX_OBSTACLES; oi = oi + 1u) {
    let obs = P.obstacles[oi];
    if (obs.w <= 0.0) {
      continue;
    }

    let delta = pos - obs.xy;
    let dist = length(delta);
    let limit = obs.z + P.h;
    if (dist < limit) {
      let n = select(vec2<f32>(0.0, 1.0), delta / max(dist, 1e-5), dist > 1e-5);
      let relVel = vel - P.obstacleVels[oi].xy;
      acc += P.boundaryStiffness * (limit - dist) * n - P.boundaryDamping * relVel;
    }
  }

  return acc;
}

fn mouseAcceleration(pos: vec2<f32>) -> vec2<f32> {
  if (P.mouseMode == 0u) {
    return vec2<f32>(0.0);
  }

  let delta = P.mousePos - pos;
  let dist = length(delta);
  if (dist >= P.mouseRadius) {
    return vec2<f32>(0.0);
  }

  let q = 1.0 - dist / P.mouseRadius;
  let w = q * q * (3.0 - 2.0 * q);
  if (P.mouseMode == 1u) {
    return P.mouseVel * P.mouseStrength * w;
  }

  let dir = select(vec2<f32>(0.0, 0.0), delta / max(dist, 1e-5), dist > 1e-5);
  return dir * (P.mouseStrength * 34.0) * w;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.count) {
    return;
  }

  let particle = particlesIn[i];
  let pos = particle.pos;
  let vel = particle.vel;
  let rhoI = max(density[i], P.restDensity * 0.1);
  let pI = pressure[i];
  let base = vec2<i32>(floor(pos / P.h));
  var acc = vec2<f32>(sin(P.tilt), -cos(P.tilt)) * P.gravity;

  for (var oy: i32 = -1; oy <= 1; oy = oy + 1) {
    let cy = base.y + oy;
    if (cy < 0 || cy >= i32(P.gridH)) {
      continue;
    }

    for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
      let cx = base.x + ox;
      if (cx < 0 || cx >= i32(P.gridW)) {
        continue;
      }

      let cell = u32(cy) * P.gridW + u32(cx);
      let start = cellStart[cell];
      let end = start + cellCount[cell];
      for (var s = start; s < end; s = s + 1u) {
        let j = sortedIndices[s];
        let other = particlesIn[j];
        let rvec = pos - other.pos;
        let r2 = dot(rvec, rvec);
        if (r2 <= 1e-12 || r2 >= P.h * P.h) {
          continue;
        }

        let r = sqrt(r2);
        let rhoJ = max(density[j], P.restDensity * 0.1);
        let pJ = pressure[j];
        let gradW = spikyGrad(rvec, r);
        let pressureTerm = pI / (rhoI * rhoI) + pJ / (rhoJ * rhoJ);
        acc += -P.mass * pressureTerm * gradW;

        let lap = viscosityLap(r);
        acc += (P.viscosity / rhoI) * (P.mass / rhoJ) * (other.vel - vel) * lap;
      }
    }
  }

  acc += wallAcceleration(pos, vel);
  acc += obstacleAcceleration(pos, vel);
  acc += mouseAcceleration(pos);

  var nextVel = (vel + P.dt * acc) * P.damping;
  if (!(nextVel.x == nextVel.x) || !(nextVel.y == nextVel.y) || abs(nextVel.x) > 1e20 || abs(nextVel.y) > 1e20) {
    nextVel = vec2<f32>(0.0);
  }
  let speed = length(nextVel);
  if (speed > P.maxSpeed) {
    nextVel = nextVel * (P.maxSpeed / speed);
  }

  var nextPos = pos + P.dt * nextVel;
  nextPos = clamp(nextPos, vec2<f32>(0.001), P.boxSize - vec2<f32>(0.001));

  particlesOut[i].pos = nextPos;
  particlesOut[i].vel = nextVel;
}
`;

export const splatWGSL = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> P: Params;

${RENDER_HELPERS}

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) local: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) instance: u32) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
  );

  let local = corners[vi];
  let center = worldToPixel(particles[instance].pos);
  let radiusPx = max(P.viewScale * P.h * 1.35, 3.0);

  var out: VSOut;
  out.clip = pixelToClip(center + local * radiusPx);
  out.local = local;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.local, in.local);
  if (r2 > 1.0) {
    discard;
  }

  let falloff = pow(1.0 - r2, 3.0);
  return vec4<f32>(falloff * 0.34, falloff * 0.18, falloff * 0.06, 1.0);
}
`;

const FULLSCREEN_VS = PARAMS_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> P: Params;

struct VSOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var pts = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  let p = pts[vi];

  var out: VSOut;
  out.clip = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
  return out;
}
`;

export const surfaceWGSL = FULLSCREEN_VS + /* wgsl */ `
@group(0) @binding(1) var fieldSampler: sampler;
@group(0) @binding(2) var fieldTex: texture_2d<f32>;

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let rawDims = textureDimensions(fieldTex);
  let dims = vec2<f32>(f32(rawDims.x), f32(rawDims.y));
  let texel = 1.0 / dims;
  let f = textureSample(fieldTex, fieldSampler, in.uv).r;
  let fx = textureSample(fieldTex, fieldSampler, in.uv + vec2<f32>(texel.x, 0.0)).r -
           textureSample(fieldTex, fieldSampler, in.uv - vec2<f32>(texel.x, 0.0)).r;
  let fy = textureSample(fieldTex, fieldSampler, in.uv + vec2<f32>(0.0, texel.y)).r -
           textureSample(fieldTex, fieldSampler, in.uv - vec2<f32>(0.0, texel.y)).r;

  let edge = smoothstep(P.iso - 0.055, P.iso + 0.055, f);
  let normal = normalize(vec3<f32>(-fx * 8.5, fy * 8.5, 1.0));
  let light = normalize(vec3<f32>(-0.45, -0.55, 0.72));
  let view = vec3<f32>(0.0, 0.0, 1.0);
  let diff = clamp(dot(normal, light), 0.0, 1.0);
  let halfVec = normalize(light + view);
  let spec = pow(clamp(dot(normal, halfVec), 0.0, 1.0), 72.0);
  let rim = pow(1.0 - clamp(normal.z, 0.0, 1.0), 2.5);
  let depth = clamp((f - P.iso) * 1.1, 0.0, 1.0);

  let bgTop = vec3<f32>(0.005, 0.010, 0.018);
  let bgBot = vec3<f32>(0.000, 0.003, 0.006);
  let bg = mix(bgBot, bgTop, in.uv.y);
  let shallow = vec3<f32>(0.12, 0.82, 1.00);
  let deep = vec3<f32>(0.005, 0.14, 0.32);
  var water = mix(shallow, deep, depth);
  water *= 0.54 + diff * 0.46;
  water += spec * vec3<f32>(0.85, 1.0, 1.0);
  water += rim * vec3<f32>(0.05, 0.34, 0.45);

  let color = mix(bg, water, edge);
  return vec4<f32>(color, 1.0);
}
`;

export const overlayWGSL = FULLSCREEN_VS + RENDER_HELPERS + /* wgsl */ `
@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let px = in.clip.xy;
  let pos = pixelToWorld(px);
  let borderPx = 2.0;
  let inset = borderPx / P.viewScale;
  let inBox = pos.x >= -inset && pos.x <= P.boxSize.x + inset &&
              pos.y >= -inset && pos.y <= P.boxSize.y + inset;
  let nearFrame = inBox && (
    abs(pos.x) <= inset ||
    abs(pos.x - P.boxSize.x) <= inset ||
    abs(pos.y) <= inset ||
    abs(pos.y - P.boxSize.y) <= inset
  );

  if (nearFrame) {
    return vec4<f32>(0.36, 0.62, 0.38, 0.82);
  }

  for (var oi = 0u; oi < MAX_OBSTACLES; oi = oi + 1u) {
    let obs = P.obstacles[oi];
    if (obs.w <= 0.0) {
      continue;
    }

    let dist = distance(pos, obs.xy);
    let edge = 1.5 / P.viewScale;
    let fill = 1.0 - smoothstep(obs.z - edge, obs.z + edge, dist);
    if (fill > 0.0) {
      // Slate disc with a bright rim so it reads as a solid object on the dark field.
      let body = vec3<f32>(0.16, 0.19, 0.24);
      let rim = smoothstep(obs.z - 1.6 / P.viewScale, obs.z, dist);
      let highlight = vec3<f32>(0.46, 0.52, 0.60);
      let color = mix(body, highlight, rim);
      return vec4<f32>(color, fill);
    }
  }

  return vec4<f32>(0.0);
}
`;
