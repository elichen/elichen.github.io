import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const B = require("../vendor/babylon.js");
globalThis.window = Object.assign(new EventTarget(), { BABYLON: B });
const { atelier, makeSam } = await import("../objects.js");

function fixture() {
  const engine = new B.NullEngine();
  const scene = new B.Scene(engine);
  return { engine, sam: makeSam(atelier(scene)) };
}

// An ellipsoid value below 1 means the flipper tip is buried in the torso.
function tipOutsideBody(sam, arm) {
  const mesh = arm.getChildMeshes()[0];
  sam.root.computeWorldMatrix(true);
  sam.body.computeWorldMatrix(true);
  arm.computeWorldMatrix(true);
  const worldTip = B.Vector3.TransformCoordinates(
    new B.Vector3(0, -0.45, 0),
    mesh.computeWorldMatrix(true),
  );
  const tip = B.Vector3.TransformCoordinates(
    worldTip,
    B.Matrix.Invert(sam.body.getWorldMatrix()),
  );
  return (
    tip.x ** 2 / 0.87 ** 2 +
    (tip.y - 1.02) ** 2 / 0.89 ** 2 +
    tip.z ** 2 / 0.71 ** 2
  );
}

test("both flipper tips stay outside the torso through greetings and release", () => {
  const { engine, sam } = fixture();
  try {
    // Tap greetings last 3 seconds; arrival greetings last 2.4 seconds.
    for (const duration of [3, 2.4]) {
      for (const walking of [false, true]) {
        const dt = 1 / 60;
        for (let frame = 0; frame < (duration + 1) / dt; frame++) {
          sam.update(
            frame * dt,
            walking,
            Math.max(0, duration - frame * dt),
            dt,
          );
          for (const [side, arm] of sam.arms.entries()) {
            assert.ok(
              tipOutsideBody(sam, arm) > 1,
              `flipper ${side} entered torso at frame ${frame}, walking=${walking}`,
            );
          }
        }
      }
    }
  } finally {
    engine.dispose();
  }
});

test("starting, retriggering, and ending a greeting do not snap the flipper", () => {
  const { engine, sam } = fixture();
  try {
    for (const remaining of [0, 3, 2.9, 2.2, 3, 0, 0, 0]) {
      const before = sam.arms[1].rotation.z;
      sam.update(0, false, remaining, 1 / 60);
      assert.ok(Math.abs(sam.arms[1].rotation.z - before) < 0.26);
    }
    for (let i = 0; i < 90; i++) sam.update(0, false, 0, 1 / 60);
    assert.ok(Math.abs(sam.arms[1].rotation.z) < 0.001);
  } finally {
    engine.dispose();
  }
});
