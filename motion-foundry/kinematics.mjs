/**
 * Analytic, planar rigid-link kinematics. Canonical ground length is 1 for a
 * four-bar; a slider has a horizontal guide at params.railOffset. The coupler
 * point is A + traceAlong(B-A) + traceOffset perp(B-A).
 *
 * A design transform is the orientation-preserving similarity
 * x' = a*x - b*y + tx, y' = b*x + a*y + ty. Link lengths below are canonical;
 * displayed lengths multiply by hypot(a,b). Angles are radians.
 */

export const TAU = Math.PI * 2;
const IDENTITY = Object.freeze({ a: 1, b: 0, tx: 0, ty: 0 });
const trigCache = new Map();
const finitePoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y);

export function transformPoint(point, transform = IDENTITY) {
  return {
    x: transform.a * point.x - transform.b * point.y + transform.tx,
    y: transform.b * point.x + transform.a * point.y + transform.ty,
  };
}

/** Return a normalized, full-cycle-valid copy, or null. No silent length edits. */
export function validateDesign(design) {
  if (
    !design ||
    !["fourbar", "slider"].includes(design.family) ||
    !design.params
  )
    return null;
  const p = design.params;
  const required =
    design.family === "fourbar"
      ? ["crank", "coupler", "rocker", "traceAlong", "traceOffset"]
      : ["crank", "coupler", "railOffset", "traceAlong", "traceOffset"];
  if (
    !required.every((key) => Number.isFinite(p[key]) && Math.abs(p[key]) <= 100)
  )
    return null;
  if (
    p.crank <= 1e-5 ||
    p.coupler <= 1e-5 ||
    (design.family === "fourbar" && p.rocker <= 1e-5)
  )
    return null;
  const branch = p.branch === undefined ? 1 : p.branch;
  if (branch !== 1 && branch !== -1) return null;
  const margin = 1e-7 * Math.max(1, p.crank, p.coupler, p.rocker || 0);
  if (design.family === "fourbar") {
    // Circle intersections exist strictly for every crank angle. These two
    // inequalities include crank-rocker and double-crank mechanisms.
    if (
      p.coupler + p.rocker <= 1 + p.crank + margin ||
      Math.abs(p.coupler - p.rocker) >= Math.abs(1 - p.crank) - margin
    )
      return null;
  } else if (p.coupler <= p.crank + Math.abs(p.railOffset) + margin)
    return null;
  const transform = { ...IDENTITY, ...design.transform };
  if (
    !Object.values(transform).every(Number.isFinite) ||
    Math.hypot(transform.a, transform.b) < 1e-8 ||
    Math.hypot(transform.a, transform.b) > 1e6
  )
    return null;
  return {
    family: design.family,
    params: Object.fromEntries(
      [...required, "branch"].map((key) => [
        key,
        key === "branch" ? branch : p[key],
      ]),
    ),
    transform: {
      a: transform.a,
      b: transform.b,
      tx: transform.tx,
      ty: transform.ty,
    },
  };
}

/** The closure calculation shared by animation and the optimizer. */
function canonicalPose(family, p, cosine, sine) {
  const ax = p.crank * cosine,
    ay = p.crank * sine;
  const branch = p.branch ?? 1;
  let bx, by;
  if (family === "fourbar") {
    const dx = 1 - ax,
      dy = -ay,
      distance = Math.hypot(dx, dy);
    if (distance < 1e-12) return null;
    const along =
      (p.coupler ** 2 - p.rocker ** 2 + distance ** 2) / (2 * distance);
    const heightSquared = p.coupler ** 2 - along ** 2;
    if (heightSquared < -1e-10) return null;
    const height = branch * Math.sqrt(Math.max(0, heightSquared));
    bx = ax + (along * dx - height * dy) / distance;
    by = ay + (along * dy + height * dx) / distance;
  } else {
    const dy = p.railOffset - ay;
    const widthSquared = p.coupler ** 2 - dy ** 2;
    if (widthSquared < -1e-10) return null;
    bx = ax + branch * Math.sqrt(Math.max(0, widthSquared));
    by = p.railOffset;
  }
  const ux = bx - ax,
    uy = by - ay;
  const px = ax + p.traceAlong * ux - p.traceOffset * uy;
  const py = ay + p.traceAlong * uy + p.traceOffset * ux;
  if (![ax, ay, bx, by, px, py].every(Number.isFinite)) return null;
  return { ax, ay, bx, by, px, py };
}

/** Return transformed joints, the attached tracer, and moving link segments. */
export function mechanismAt(design, theta) {
  const invalid = { valid: false, joints: {}, tracer: null, links: [] };
  if (
    !design?.params ||
    !Number.isFinite(theta) ||
    !["fourbar", "slider"].includes(design.family)
  )
    return invalid;
  const raw = canonicalPose(
    design.family,
    design.params,
    Math.cos(theta),
    Math.sin(theta),
  );
  if (!raw) return invalid;
  const transform = { ...IDENTITY, ...design.transform };
  const point = (x, y) => transformPoint({ x, y }, transform);
  const O = point(0, 0),
    A = point(raw.ax, raw.ay),
    B = point(raw.bx, raw.by);
  const G = point(1, design.family === "slider" ? design.params.railOffset : 0);
  const tracer = point(raw.px, raw.py);
  if (![O, A, B, G, tracer].every(finitePoint)) return invalid;
  const pose = {
    valid: true,
    joints: { O, A, B, G },
    tracer,
    links:
      design.family === "fourbar"
        ? [
            [O, A],
            [A, B],
            [B, G],
          ]
        : [
            [O, A],
            [A, B],
          ],
  };
  if (design.family === "slider") {
    const p = design.params;
    const low = Math.sqrt(
      Math.max(0, (p.coupler - p.crank) ** 2 - p.railOffset ** 2),
    );
    const high = Math.sqrt(
      Math.max(0, (p.coupler + p.crank) ** 2 - p.railOffset ** 2),
    );
    const sign = p.branch ?? 1;
    pose.guide = [
      point((sign > 0 ? low : -high) - 0.25, p.railOffset),
      point((sign > 0 ? high : -low) + 0.25, p.railOffset),
    ];
  }
  return pose;
}

/** Flat joint aliases are convenient for the drawing surface. */
export function poseAt(design, theta) {
  const pose = mechanismAt(design, theta);
  return { ...pose, ...pose.joints, P: pose.tracer };
}

/** Packed coordinates for the synthesis engine; samples are constant crank speed. */
export function sampleCoordinates(design, count = 128, transformed = true) {
  if (
    !design?.params ||
    !["fourbar", "slider"].includes(design.family) ||
    !Number.isFinite(count)
  )
    return null;
  count = Math.max(8, Math.min(2048, Math.round(count)));
  let trig = trigCache.get(count);
  if (!trig) {
    trig = Array.from({ length: count }, (_, i) => [
      Math.cos((TAU * i) / count),
      Math.sin((TAU * i) / count),
    ]);
    if (trigCache.size > 12) trigCache.clear();
    trigCache.set(count, trig);
  }
  const out = new Float64Array(count * 2);
  const transform = transformed
    ? { ...IDENTITY, ...design.transform }
    : IDENTITY;
  for (let i = 0; i < count; i++) {
    const pose = canonicalPose(
      design.family,
      design.params,
      trig[i][0],
      trig[i][1],
    );
    if (!pose) return null;
    out[i * 2] = transform.a * pose.px - transform.b * pose.py + transform.tx;
    out[i * 2 + 1] =
      transform.b * pose.px + transform.a * pose.py + transform.ty;
  }
  return out;
}

export function sampleMechanism(design, count = 128) {
  const packed = sampleCoordinates(design, count);
  return packed
    ? Array.from({ length: packed.length / 2 }, (_, i) => ({
        x: packed[i * 2],
        y: packed[i * 2 + 1],
      }))
    : [];
}

/** Resample a closed polyline, including its final-to-first segment, by distance. */
export function resampleClosed(points, count = 96) {
  if (!Array.isArray(points) || points.length < 3 || !points.every(finitePoint))
    return [];
  const packed = Float64Array.from(points.flatMap((p) => [p.x, p.y]));
  const sampled = resampleCoordinates(packed, count);
  return sampled
    ? Array.from({ length: sampled.length / 2 }, (_, i) => ({
        x: sampled[i * 2],
        y: sampled[i * 2 + 1],
      }))
    : [];
}

export function resampleCoordinates(points, count = 64) {
  if (!points) return null;
  const length = points.length / 2;
  if (length < 3 || !Number.isFinite(count)) return null;
  count = Math.max(8, Math.min(2048, Math.round(count)));
  const cumulative = new Float64Array(length + 1);
  for (let i = 0; i < length; i++) {
    const next = (i + 1) % length;
    cumulative[i + 1] =
      cumulative[i] +
      Math.hypot(
        points[next * 2] - points[i * 2],
        points[next * 2 + 1] - points[i * 2 + 1],
      );
  }
  const perimeter = cumulative[length];
  if (!Number.isFinite(perimeter) || perimeter < 1e-9) return null;
  const out = new Float64Array(count * 2);
  let segment = 0;
  for (let i = 0; i < count; i++) {
    const distance = (perimeter * i) / count;
    while (segment < length - 1 && cumulative[segment + 1] <= distance)
      segment++;
    const next = (segment + 1) % length;
    const segmentLength = cumulative[segment + 1] - cumulative[segment];
    const t =
      segmentLength > 1e-12
        ? (distance - cumulative[segment]) / segmentLength
        : 0;
    out[i * 2] = points[segment * 2] * (1 - t) + points[next * 2] * t;
    out[i * 2 + 1] =
      points[segment * 2 + 1] * (1 - t) + points[next * 2 + 1] * t;
  }
  return out;
}

/** Useful inspectable starting mechanisms. Preset targets are independent shapes. */
export function presetDesign(name = "stride") {
  if (name === "oval" || name === "slider")
    return {
      family: "slider",
      params: {
        crank: 1,
        coupler: 2.8,
        railOffset: 0.2,
        traceAlong: 0.6,
        traceOffset: 0.8,
        branch: 1,
      },
      transform: { ...IDENTITY },
    };
  return {
    family: "fourbar",
    params: {
      crank: name === "eight" ? 1.2 : 0.38,
      coupler: name === "eight" ? 1.15 : 1.18,
      rocker: name === "eight" ? 1.18 : 0.94,
      traceAlong: name === "eight" ? 0.55 : 1.25,
      traceOffset: name === "eight" ? 0.35 : 0.48,
      branch: 1,
    },
    transform: { ...IDENTITY },
  };
}

/** Drawn target examples, deliberately not baked-in answers to the search. */
export function presetTarget(name = "stride", count = 96) {
  const points = Array.from({ length: 256 }, (_, i) => {
    const theta = (TAU * i) / 256;
    const c = Math.cos(theta),
      s = Math.sin(theta);
    if (name === "oval") return { x: 0.84 * c, y: 0.44 * s };
    if (name === "eight") return { x: 0.82 * s, y: 0.36 * Math.sin(2 * theta) };
    if (name === "petal") {
      const radius = 0.65 + 0.13 * Math.cos(3 * theta);
      return { x: radius * c, y: radius * s };
    }
    return {
      x: 0.86 * c + 0.07 * Math.sin(2 * theta),
      y: 0.24 * s + 0.14 * s * s - 0.12,
    };
  });
  return resampleClosed(points, count);
}
