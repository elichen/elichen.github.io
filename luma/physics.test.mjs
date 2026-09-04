import test from "node:test";
import assert from "node:assert/strict";
import {
  reflect,
  refract,
  direction,
  dot,
  mul,
  sub,
  unit,
  intersectSegment,
  lensDirection,
  traceRay,
  traceScene,
  PRESETS,
} from "./physics.mjs";

const near = (actual, expected, epsilon = 1e-8) =>
  assert.ok(Math.abs(actual - expected) < epsilon, `${actual} ≈ ${expected}`);
const nearVector = (actual, expected) => {
  near(actual.x, expected.x);
  near(actual.y, expected.y);
};

test("segment intersections reject parallel, behind, and out-of-segment hits", () => {
  const ray = { x: 1, y: 0 },
    origin = { x: 0, y: 0 };
  near(intersectSegment(origin, ray, { x: 10, y: -2 }, { x: 10, y: 2 }).t, 10);
  assert.equal(
    intersectSegment(origin, ray, { x: -10, y: -2 }, { x: -10, y: 2 }),
    null,
  );
  assert.equal(
    intersectSegment(origin, ray, { x: 2, y: 1 }, { x: 5, y: 1 }),
    null,
  );
  assert.equal(
    intersectSegment(origin, ray, { x: 10, y: 2 }, { x: 10, y: 4 }),
    null,
  );
});

test("mirror preserves the angle to the normal and reflects reversibly", () => {
  for (let a = -80; a <= 80; a += 10) {
    const incident = direction(a),
      normal = { x: -1, y: 0 },
      reflected = reflect(incident, normal);
    near(dot(incident, normal), -dot(reflected, normal));
    nearVector(reflect(reflected, normal), incident);
    near(Math.hypot(reflected.x, reflected.y), 1);
  }
});

test("Snell refraction satisfies n1 sin(theta1) = n2 sin(theta2) and is reversible", () => {
  for (let a = 0; a <= 80; a += 10) {
    const incident = direction(a),
      normal = { x: -1, y: 0 };
    const result = refract(incident, normal, 1, 1.52);
    assert.equal(result.reflected, false);
    near(incident.y, 1.52 * result.ray.y);
    nearVector(
      refract(mul(result.ray, -1), normal, 1.52, 1).ray,
      mul(incident, -1),
    );
  }
});

test("a glass-to-air ray above the critical angle reflects internally", () => {
  const normal = { x: -1, y: 0 },
    incident = direction(50);
  assert.equal(refract(direction(30), normal, 1.5, 1).reflected, false);
  const result = refract(incident, normal, 1.5, 1);
  assert.equal(result.reflected, true);
  nearVector(result.ray, reflect(incident, normal));
});

test("parallel rays converge at a lens focus from either side and at any rotation", () => {
  for (const angle of [0, 37, 90, -80])
    for (const side of [-1, 1]) {
      const lens = { x: 600, y: 380, angle, focal: 240 };
      const axis = direction(angle),
        tangent = direction(angle + 90);
      for (const height of [-90, -30, 0, 60, 90]) {
        const hit = {
          x: lens.x + tangent.x * height,
          y: lens.y + tangent.y * height,
        };
        const outgoing = lensDirection(mul(axis, side), hit, lens);
        const focus = {
          x: lens.x + side * axis.x * lens.focal,
          y: lens.y + side * axis.y * lens.focal,
        };
        nearVector(outgoing, unit(sub(focus, hit)));
      }
    }
});

test("a central lens ray is undeviated and oblique rays obey the paraxial rule", () => {
  const lens = { x: 600, y: 380, angle: 0, focal: 240 };
  nearVector(lensDirection(direction(12), lens, lens), direction(12));
  const result = lensDirection(direction(12), { x: 600, y: 410 }, lens);
  near(result.y / result.x, Math.tan((12 * Math.PI) / 180) - 30 / 240);
});

test("the prism preset transmits each ray through two surfaces and disperses violet more than red", () => {
  const paths = traceScene(PRESETS.prism.objects);
  assert.equal(paths.length, 287);
  assert.ok(paths.every((p) => p.segments.length === 3));
  const violet = paths[0].segments.at(-1),
    red = paths[40].segments.at(-1);
  const angle = (s) => Math.atan2(s.b.y - s.a.y, s.b.x - s.a.x);
  assert.ok(angle(violet) > angle(red) + 0.1);
});

test("the mirror preset routes the beam up, then right, and through the prism", () => {
  const segments = traceScene(PRESETS.mirrors.objects)[3].segments;
  assert.equal(segments.length, 5);
  near(segments[1].a.x, segments[1].b.x);
  assert.ok(segments[1].b.y < segments[1].a.y);
  near(segments[2].a.y, segments[2].b.y);
  assert.ok(segments[2].b.x > segments[2].a.x);
});

test("mirror cavities terminate at the bounce budget and all presets remain finite", () => {
  const mirrors = [
    { type: "mirror", x: 300, y: 380, angle: 0, size: 300 },
    { type: "mirror", x: 800, y: 380, angle: 0, size: 300 },
  ];
  assert.equal(
    traceRay({ x: 500, y: 380 }, direction(0), 550, mirrors, 24).length,
    24,
  );
  for (const preset of Object.values(PRESETS))
    for (const path of traceScene(preset.objects))
      for (const s of path.segments) {
        assert.ok(
          [s.a.x, s.a.y, s.b.x, s.b.y, s.energy].every(Number.isFinite),
        );
        assert.ok(s.energy > 0 && s.energy <= 1);
      }
});

test("disabled sources emit no light and an empty scene is valid", () => {
  assert.deepEqual(traceScene([]), []);
  assert.deepEqual(
    traceScene([{ ...PRESETS.prism.objects[0], enabled: false }]),
    [],
  );
});

test("the crossing preset sends two spectra across each other within the workbench", () => {
  const paths = traceScene(PRESETS.split.objects);
  assert.ok(paths.every((path) => path.segments.length === 3));
  const upper = paths[20].segments.at(-1);
  const lower = paths[307].segments.at(-1);
  const hit = intersectSegment(
    upper.a,
    unit(sub(upper.b, upper.a)),
    lower.a,
    lower.b,
  );
  assert.ok(hit);
  assert.ok(hit.point.x > 600 && hit.point.x < 1200);
  assert.ok(hit.point.y > 300 && hit.point.y < 460);
});
