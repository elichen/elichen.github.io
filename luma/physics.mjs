// Geometric optics in a two-dimensional, 1200 × 760 world.
export const WORLD = { width: 1200, height: 760 };
export const EPS = 1e-5;
export const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
export const mul = (a, s) => ({ x: a.x * s, y: a.y * s });
export const dot = (a, b) => a.x * b.x + a.y * b.y;
export const cross = (a, b) => a.x * b.y - a.y * b.x;
export const unit = (a) => mul(a, 1 / (Math.hypot(a.x, a.y) || 1));
export const direction = (degrees) => ({
  x: Math.cos((degrees * Math.PI) / 180),
  y: Math.sin((degrees * Math.PI) / 180),
});
export const rotate = (p, degrees) => {
  const d = direction(degrees);
  return { x: p.x * d.x - p.y * d.y, y: p.x * d.y + p.y * d.x };
};
export function vertices(object) {
  return [0, 120, 240].map((a) =>
    add(object, mul(direction(a + object.angle - 90), object.size)),
  );
}
export function endpoints(object) {
  const d = mul(direction(object.angle + 90), object.size / 2);
  return [sub(object, d), add(object, d)];
}
export function contains(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i],
      b = polygon[j];
    if (
      a.y > point.y !== b.y > point.y &&
      point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x
    )
      inside = !inside;
  }
  return inside;
}
export function intersectSegment(origin, ray, a, b) {
  const edge = sub(b, a),
    denominator = cross(ray, edge);
  if (Math.abs(denominator) < EPS) return null;
  const offset = sub(a, origin);
  const t = cross(offset, edge) / denominator,
    u = cross(offset, ray) / denominator;
  if (t <= EPS || u < -EPS || u > 1 + EPS) return null;
  return {
    t,
    point: add(origin, mul(ray, t)),
    normal: unit({ x: -edge.y, y: edge.x }),
  };
}
export function reflect(ray, normal) {
  return unit(sub(ray, mul(normal, 2 * dot(ray, normal))));
}
export function refract(ray, normal, n1, n2) {
  // Orient the normal against the incident ray, then apply vector Snell's law.
  if (dot(ray, normal) > 0) normal = mul(normal, -1);
  const cosine = -dot(normal, ray),
    ratio = n1 / n2;
  const discriminant = 1 - ratio * ratio * (1 - cosine * cosine);
  if (discriminant < 0) return { ray: reflect(ray, normal), reflected: true };
  return {
    ray: unit(
      add(
        mul(ray, ratio),
        mul(normal, ratio * cosine - Math.sqrt(discriminant)),
      ),
    ),
    reflected: false,
  };
}
export function glassIndex(object, wavelength) {
  // Cauchy-style dispersion, referenced at 550 nm. This is an illustrative
  // material, not a fit to a particular commercial optical glass.
  return (
    1.52 + object.dispersion * (1 / (wavelength / 1000) ** 2 - 1 / 0.55 ** 2)
  );
}
export function lensDirection(ray, point, lens) {
  const axis = direction(lens.angle),
    tangent = direction(lens.angle + 90);
  const axial = dot(ray, axis);
  if (Math.abs(axial) < EPS) return ray;
  const sign = Math.sign(axial),
    height = dot(sub(point, lens), tangent);
  const slope = dot(ray, tangent) / Math.abs(axial) - height / lens.focal;
  return unit(add(mul(axis, sign), mul(tangent, slope)));
}
export function wavelengthRGB(wavelength) {
  let r = 0,
    g = 0,
    b = 0;
  if (wavelength < 440) {
    r = (440 - wavelength) / 60;
    b = 1;
  } else if (wavelength < 490) {
    g = (wavelength - 440) / 50;
    b = 1;
  } else if (wavelength < 510) {
    g = 1;
    b = (510 - wavelength) / 20;
  } else if (wavelength < 580) {
    r = (wavelength - 510) / 70;
    g = 1;
  } else if (wavelength < 645) {
    r = 1;
    g = (645 - wavelength) / 65;
  } else r = 1;
  return [r, g, b].map((v) =>
    Math.round(255 * Math.max(0, Math.min(1, v)) ** 0.8),
  );
}
function mediumAt(point, prisms, wavelength) {
  const medium = prisms.findLast((p) => contains(point, p.polygon));
  return medium ? glassIndex(medium.object, wavelength) : 1;
}
export function traceRay(origin, ray, wavelength, objects, maxBounces = 24) {
  const surfaces = [],
    prisms = [];
  for (const object of objects) {
    if (object.type === "source") continue;
    const points =
      object.type === "prism" ? vertices(object) : endpoints(object);
    if (object.type === "prism") prisms.push({ object, polygon: points });
    for (let i = 0; i < (object.type === "prism" ? 3 : 1); i++) {
      surfaces.push({
        object,
        a: points[i],
        b: points[(i + 1) % points.length],
      });
    }
  }
  const bounds = [
    { x: 0, y: 0 },
    { x: WORLD.width, y: 0 },
    { x: WORLD.width, y: WORLD.height },
    { x: 0, y: WORLD.height },
  ];
  for (let i = 0; i < 4; i++)
    surfaces.push({ a: bounds[i], b: bounds[(i + 1) % 4], object: null });
  const segments = [];
  let energy = 1;
  for (let bounce = 0; bounce < maxBounces; bounce++) {
    let nearest = null;
    for (const surface of surfaces) {
      const hit = intersectSegment(origin, ray, surface.a, surface.b);
      if (hit && (!nearest || hit.t < nearest.t))
        nearest = { ...hit, object: surface.object };
    }
    if (!nearest) break;
    segments.push({ a: origin, b: nearest.point, energy, wavelength });
    const object = nearest.object;
    if (!object) break;
    if (object.type === "mirror") {
      ray = reflect(ray, nearest.normal);
      energy *= 0.96;
    } else if (object.type === "prism") {
      const n1 = mediumAt(
        sub(nearest.point, mul(ray, EPS * 10)),
        prisms,
        wavelength,
      );
      const n2 = mediumAt(
        add(nearest.point, mul(ray, EPS * 10)),
        prisms,
        wavelength,
      );
      const result = refract(ray, nearest.normal, n1, n2);
      ray = result.ray;
      if (!result.reflected) energy *= 0.97;
    } else if (object.type === "lens") {
      ray = lensDirection(ray, nearest.point, object);
      energy *= 0.98;
    }
    origin = add(nearest.point, mul(ray, EPS * 20));
  }
  return segments;
}
export function traceScene(objects) {
  const paths = [];
  for (const source of objects.filter(
    (o) => o.type === "source" && o.enabled !== false,
  )) {
    const wavelengths =
      source.color === "white"
        ? Array.from({ length: 41 }, (_, i) => 390 + i * 7.5)
        : [Number(source.color)];
    const rays = source.width > 2 ? 7 : 1;
    for (let i = 0; i < rays; i++) {
      const offset = rays === 1 ? 0 : (i / (rays - 1) - 0.5) * source.width;
      const origin = add(
        add(source, mul(direction(source.angle), 29)),
        mul(direction(source.angle + 90), offset),
      );
      for (const wavelength of wavelengths) {
        paths.push({
          wavelength,
          white: source.color === "white",
          segments: traceRay(
            origin,
            direction(source.angle),
            wavelength,
            objects,
          ),
        });
      }
    }
  }
  return paths;
}
export const PRESETS = {
  prism: {
    name: "Prism study",
    caption: "One beam. A hidden spectrum.",
    objects: [
      {
        id: 1,
        type: "source",
        x: 180,
        y: 355,
        angle: -12,
        width: 12,
        color: "white",
        enabled: true,
      },
      {
        id: 2,
        type: "prism",
        x: 565,
        y: 330,
        angle: 0,
        size: 150,
        dispersion: 0.025,
      },
    ],
  },
  mirrors: {
    name: "Around the bend",
    caption: "A little geometry goes a long way.",
    objects: [
      {
        id: 1,
        type: "source",
        x: 150,
        y: 510,
        angle: 0,
        width: 10,
        color: "530",
        enabled: true,
      },
      { id: 2, type: "mirror", x: 430, y: 510, angle: 45, size: 170 },
      { id: 3, type: "mirror", x: 430, y: 235, angle: 45, size: 170 },
      {
        id: 4,
        type: "prism",
        x: 770,
        y: 235,
        angle: 15,
        size: 130,
        dispersion: 0.025,
      },
    ],
  },
  focus: {
    name: "A point of focus",
    caption: "Many paths. One meeting place.",
    objects: [
      {
        id: 1,
        type: "source",
        x: 170,
        y: 365,
        angle: 0,
        width: 190,
        color: "490",
        enabled: true,
      },
      { id: 2, type: "lens", x: 520, y: 365, angle: 0, size: 290, focal: 270 },
    ],
  },
  split: {
    name: "Chromatic crossing",
    caption: "Make something beautifully tangled.",
    objects: [
      {
        id: 1,
        type: "source",
        x: 135,
        y: 265,
        angle: -12,
        width: 7,
        color: "white",
        enabled: true,
      },
      {
        id: 2,
        type: "source",
        x: 135,
        y: 495,
        angle: 12,
        width: 7,
        color: "white",
        enabled: true,
      },
      {
        id: 3,
        type: "prism",
        x: 530,
        y: 240,
        angle: 0,
        size: 125,
        dispersion: 0.03,
      },
      {
        id: 4,
        type: "prism",
        x: 530,
        y: 520,
        angle: 180,
        size: 125,
        dispersion: 0.03,
      },
    ],
  },
};
