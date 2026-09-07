import * as THREE from "three";

const sphereGeometry = new THREE.SphereGeometry(1, 48, 32);
const lowSphere = new THREE.SphereGeometry(1, 20, 14);
export function material(color, options = {}) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.38,
    metalness: 0,
    ...options,
  });
}
export function ball(parent, mat, x, y, z, sx, sy = sx, sz = sx, low = false) {
  const mesh = new THREE.Mesh(low ? lowSphere : sphereGeometry, mat);
  mesh.position.set(x, y, z);
  mesh.scale.set(sx, sy, sz);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}
export function tube(parent, points, radius, mat, closed = false) {
  const curve = new THREE.CatmullRomCurve3(
    points.map((p) => (p.isVector3 ? p : new THREE.Vector3(...p))),
    closed,
  );
  const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(
      curve,
      Math.max(24, points.length * 5),
      radius,
      8,
      closed,
    ),
    mat,
  );
  mesh.castShadow = true;
  parent.add(mesh);
  return mesh;
}
export function cylinder(
  parent,
  mat,
  top,
  bottom,
  height,
  x,
  y,
  z,
  segments = 32,
) {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(top, bottom, height, segments),
    mat,
  );
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}
export function box(parent, mat, x, y, z, sx, sy, sz) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), mat);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}
export function makeDoraemon() {
  const root = new THREE.Group();
  const blue = material("#009cdb", { roughness: 0.48 });
  const white = material("#fffef6", { roughness: 0.52 });
  const ink = material("#182b32", { roughness: 0.8 });
  const red = material("#e93236", { roughness: 0.3 });
  const gold = material("#f2c13d", { metalness: 0.36, roughness: 0.3 });
  const bamboo = material("#f1cb76", { roughness: 0.46 });
  const bambooEdge = material("#bd933f", { roughness: 0.5 });
  const mouthMat = material("#7e1e22", { roughness: 0.95 });
  const tongueMat = material("#ee6562", { roughness: 0.8 });

  // The neck is the suspension point. The body trails behind it in forward
  // flight, while the head and Take-copter stay upright (see REFERENCES.md).
  const neckHeight = 0.76;
  const flightRig = new THREE.Group();
  flightRig.position.y = neckHeight;
  root.add(flightRig);
  const torso = new THREE.Group();
  torso.position.y = -neckHeight;
  flightRig.add(torso);
  const headPivot = new THREE.Group();
  flightRig.add(headPivot);
  const head = new THREE.Group();
  head.position.y = -neckHeight;
  headPivot.add(head);

  // A short, broad torso, with two tiny leg stubs rather than a tall pear shape.
  ball(torso, blue, 0, 0.02, 0, 1.17, 1.04, 0.96);
  ball(torso, white, 0, -0.04, 0.8, 0.86, 0.77, 0.245);
  ball(head, blue, 0, 1.99, 0, 1.63, 1.55, 1.43);

  const faceZ = (x, y, lift = 0.03) =>
    1.43 *
      Math.sqrt(
        Math.max(0.012, 1 - (x / 1.63) ** 2 - ((y - 1.99) / 1.55) ** 2),
      ) +
    lift;
  // A conforming white mask gives one continuous round face, without the
  // intersecting cheek spheres and deep seams of the first version.
  function ellipsePatch(parent, mat, cx, cy, rx, ry, lift) {
    const vertices = [],
      indices = [],
      segments = 80,
      rings = 18;
    for (let j = 0; j <= rings; j++)
      for (let i = 0; i <= segments; i++) {
        const a = (i / segments) * Math.PI * 2,
          r = j / rings;
        const x = cx + Math.cos(a) * rx * r,
          y = cy + Math.sin(a) * ry * r;
        vertices.push(x, y, faceZ(x, y, lift));
      }
    for (let j = 0; j < rings; j++)
      for (let i = 0; i < segments; i++) {
        const a = j * (segments + 1) + i;
        indices.push(
          a,
          a + segments + 1,
          a + 1,
          a + 1,
          a + segments + 1,
          a + segments + 2,
        );
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }
  ellipsePatch(head, white, 0, 1.76, 1.415, 1.245, 0.025);

  // The familiar broad, almost level upper lip and round open smile.
  const vertices = [],
    indices = [],
    columns = 64,
    rows = 22;
  for (let j = 0; j <= rows; j++)
    for (let i = 0; i <= columns; i++) {
      const t = i / columns,
        x = (t - 0.5) * 2.11;
      const top = 1.82 - 0.035 * Math.sin(t * Math.PI);
      const bottom = 1.82 - 0.96 * Math.sin(t * Math.PI);
      const y = THREE.MathUtils.lerp(top, bottom, j / rows);
      vertices.push(x, y, faceZ(x, y, 0.049));
    }
  for (let j = 0; j < rows; j++)
    for (let i = 0; i < columns; i++) {
      const a = j * (columns + 1) + i;
      indices.push(
        a,
        a + columns + 1,
        a + 1,
        a + 1,
        a + columns + 1,
        a + columns + 2,
      );
    }
  const smileGeo = new THREE.BufferGeometry();
  smileGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3),
  );
  smileGeo.setIndex(indices);
  smileGeo.computeVertexNormals();
  head.add(new THREE.Mesh(smileGeo, mouthMat));
  ellipsePatch(head, tongueMat, 0, 1.066, 0.49, 0.19, 0.066);
  for (const lower of [false, true]) {
    const points = [];
    for (let i = 0; i <= 40; i++) {
      const t = i / 40,
        x = (t - 0.5) * 2.11;
      const y = 1.82 - (lower ? 0.96 : 0.035) * Math.sin(t * Math.PI);
      points.push([x, y, faceZ(x, y, 0.068)]);
    }
    tube(head, points, 0.014, ink);
  }
  tube(
    head,
    [
      [0, 2.53, faceZ(0, 2.53, 0.054)],
      [0, 2.15, faceZ(0, 2.15, 0.054)],
      [0, 1.785, faceZ(0, 1.785, 0.067)],
    ],
    0.018,
    ink,
  );

  // Touching oval eyes, smaller pupils, and restrained outlines.
  const eyes = [],
    pupils = [];
  for (const side of [-1, 1]) {
    const eye = new THREE.Group();
    eye.position.set(side * 0.316, 2.93, 1.115);
    head.add(eye);
    eyes.push(eye);
    ball(eye, ink, 0, 0, 0, 0.327, 0.447, 0.117);
    ball(eye, white, 0, 0, 0.012, 0.319, 0.439, 0.12);
    const pupil = ball(
      eye,
      ink,
      -side * 0.064,
      -0.088,
      0.126,
      0.074,
      0.112,
      0.021,
    );
    pupils.push(pupil);
    ball(pupil, white, -0.2, 0.32, 0.83, 0.23, 0.19, 0.22);
    eye.traverse((part) => {
      if (part.isMesh) {
        part.castShadow = false;
        part.receiveShadow = false;
      }
    });
    for (let i = 0; i < 3; i++) {
      const points = [];
      for (let j = 0; j <= 14; j++) {
        const t = j / 14,
          x = side * (0.53 + t * 0.74);
        const y = 2.3 + (i - 1) * (0.15 + t * 0.15);
        points.push([x, y, faceZ(x, y, 0.057)]);
      }
      tube(head, points, 0.016, ink);
    }
  }
  ball(head, red, 0, 2.56, 1.405, 0.215, 0.202, 0.19);
  ball(head, white, -0.058, 2.621, 1.568, 0.048, 0.038, 0.012);

  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(1.0, 0.076, 16, 80),
    red,
  );
  collar.rotation.x = Math.PI / 2;
  collar.scale.y = 0.9;
  collar.position.y = 0.76;
  head.add(collar);
  const bell = new THREE.Group();
  bell.position.set(0, 0.54, 1.025);
  head.add(bell);
  ball(bell, gold, 0, 0, 0, 0.23, 0.219, 0.21);
  const bellBand = new THREE.Mesh(
    new THREE.TorusGeometry(0.219, 0.018, 10, 48),
    gold,
  );
  bellBand.rotation.x = Math.PI / 2;
  bellBand.position.y = 0.034;
  bell.add(bellBand);
  tube(
    bell,
    [
      [-0.19, 0.024, 0.12],
      [0, 0.024, 0.216],
      [0.19, 0.024, 0.12],
    ],
    0.012,
    bambooEdge,
  );
  ball(bell, ink, 0, -0.06, 0.207, 0.046, 0.043, 0.009);
  tube(
    bell,
    [
      [0, -0.08, 0.211],
      [0, -0.179, 0.117],
    ],
    0.013,
    ink,
  );

  const pocket = [];
  const bellyZ = (x, y) =>
    0.8 +
    0.245 *
      Math.sqrt(
        Math.max(0.02, 1 - (x / 0.86) ** 2 - ((y + 0.04) / 0.77) ** 2),
      ) +
    0.019;
  for (let i = 0; i <= 40; i++) {
    const a = (i / 40) * Math.PI,
      x = 0.59 * Math.cos(a),
      y = -0.07 - 0.43 * Math.sin(a);
    pocket.push([x, y, bellyZ(x, y)]);
  }
  tube(torso, pocket, 0.014, ink);
  tube(
    torso,
    [
      [-0.59, -0.07, bellyZ(-0.59, -0.07)],
      [0, -0.07, bellyZ(0, -0.07)],
      [0.59, -0.07, bellyZ(0.59, -0.07)],
    ],
    0.014,
    ink,
  );

  const arms = [],
    feet = [];
  for (const side of [-1, 1]) {
    const arm = new THREE.Group();
    arm.position.set(side * 0.99, 0.41, 0.02);
    torso.add(arm);
    arms.push(arm);
    const sleeve = ball(arm, blue, side * 0.3, -0.015, 0.03, 0.47, 0.29, 0.3);
    sleeve.rotation.z = side * -0.08;
    ball(arm, white, side * 0.66, -0.03, 0.055, 0.286, 0.292, 0.285);
    const leg = new THREE.Group();
    leg.position.set(side * 0.55, -0.73, 0.015);
    torso.add(leg);
    feet.push(leg);
    const stub = cylinder(leg, blue, 0.41, 0.44, 0.46, 0, -0.07, -0.015, 48);
    stub.scale.z = 1.17;
    stub.receiveShadow = false;
    ball(leg, white, 0, -0.36, 0.16, 0.51, 0.25, 0.61);
  }
  tube(
    torso,
    [
      [0, -0.38, -0.86],
      [0, -0.37, -1.12],
      [0, -0.28, -1.22],
    ],
    0.05,
    red,
  );
  const tailAnchor = ball(torso, red, 0, -0.25, -1.25, 0.205);

  const copter = new THREE.Group();
  copter.position.set(0, 3.52, 0);
  head.add(copter);
  ball(copter, bambooEdge, 0, 0.01, 0, 0.23, 0.055, 0.22);
  cylinder(copter, bamboo, 0.038, 0.046, 0.57, 0, 0.3, 0);
  cylinder(copter, gold, 0.087, 0.087, 0.08, 0, 0.6, 0);
  const rotor = new THREE.Group();
  rotor.position.y = 0.665;
  copter.add(rotor);
  for (const side of [-1, 1]) {
    const shape = new THREE.Shape();
    shape.moveTo(0.04, -0.075);
    shape.bezierCurveTo(0.45, -0.105, 1.25, -0.2, 1.4, -0.13);
    shape.quadraticCurveTo(1.5, 0, 1.38, 0.095);
    shape.bezierCurveTo(0.95, 0.16, 0.35, 0.11, 0.04, 0.075);
    shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, {
      depth: 0.029,
      bevelEnabled: true,
      bevelSegments: 2,
      steps: 1,
      bevelSize: 0.013,
      bevelThickness: 0.012,
      curveSegments: 16,
    });
    geo.rotateX(-Math.PI / 2);
    const blade = new THREE.Mesh(geo, bamboo);
    blade.rotation.y = side === 1 ? 0 : Math.PI;
    blade.castShadow = true;
    rotor.add(blade);
    tube(
      rotor,
      [
        [side * 0.26, 0.045, 0],
        [side * 0.76, 0.047, -side * 0.04],
        [side * 1.27, 0.046, -side * 0.035],
      ],
      0.005,
      bambooEdge,
    );
  }
  ball(rotor, gold, 0, 0.035, 0, 0.098, 0.056, 0.098);
  const ghostMaterial = new THREE.MeshBasicMaterial({
    color: "#edce89",
    transparent: true,
    opacity: 0.1,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const blades = rotor.children.filter(
    (c) => c.geometry?.type === "ExtrudeGeometry",
  );
  for (const blade of blades)
    for (const angle of [Math.PI / 3, (Math.PI * 2) / 3]) {
      const ghost = new THREE.Mesh(blade.geometry, ghostMaterial);
      ghost.rotation.copy(blade.rotation);
      ghost.rotation.y += angle;
      rotor.add(ghost);
    }
  const blurMat = new THREE.MeshBasicMaterial({
    color: "#f5db9f",
    transparent: true,
    opacity: 0.04,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const blur = new THREE.Mesh(new THREE.RingGeometry(0.17, 1.45, 80), blurMat);
  blur.rotation.x = -Math.PI / 2;
  rotor.add(blur);

  let elapsed = 0,
    blinkAt = 2.8,
    waveUntil = 0,
    posePitch = 0;
  const wink = () => {
    blinkAt = elapsed + 0.01;
  };
  return {
    root,
    head,
    rotor,
    tailAnchor,
    arms,
    bell,
    wink,
    get pose() {
      return {
        bodyPitch: posePitch,
        headPitch: posePitch + headPivot.rotation.x,
      };
    },
    greet() {
      waveUntil = elapsed + 2.6;
      blinkAt = elapsed + 0.25;
    },
    update(
      time,
      dt,
      {
        speed = 0,
        flying = false,
        reduced = false,
        gaze = { x: 0, y: 0 },
      } = {},
    ) {
      elapsed = time;
      const glide = flying ? THREE.MathUtils.smoothstep(speed, 0.7, 10) : 0;
      const blend = 1 - Math.exp(-dt * 4);
      posePitch = THREE.MathUtils.lerp(posePitch, glide * 0.9, blend);
      flightRig.rotation.x = posePitch;
      headPivot.rotation.x = -posePitch * 0.98;
      headPivot.rotation.z = reduced ? 0 : Math.sin(time * 0.73) * 0.018;
      rotor.rotation.y += dt * (reduced ? 9 : 24 + speed * 1.4);
      blurMat.opacity = flying ? 0.08 : 0.035;
      bell.rotation.x = Math.sin(time * 2.2) * 0.07;
      arms.forEach((arm, i) => {
        const side = i === 0 ? -1 : 1;
        arm.rotation.z =
          side *
          (-0.24 +
            glide * 0.38 +
            (reduced ? 0 : Math.sin(time * 1.6 + i * 0.7) * 0.025));
        arm.rotation.y = -side * glide * 0.18;
        arm.rotation.x = -posePitch * 0.35;
      });
      feet.forEach((leg, i) => {
        leg.rotation.x = -glide * 0.28;
        leg.rotation.z = (i === 0 ? -1 : 1) * glide * 0.1;
        leg.position.y =
          -0.73 +
          (flying && !reduced ? Math.sin(time * 1.8 + i * 0.9) * 0.035 : 0);
      });
      if (time > blinkAt + 0.17) blinkAt = time + 3.3 + Math.random() * 3;
      const blink =
        time >= blinkAt
          ? Math.max(0.08, Math.abs(((time - blinkAt) / 0.17) * 2 - 1))
          : 1;
      eyes.forEach((eye) => (eye.scale.y = blink));
      pupils.forEach((p, i) => {
        p.position.x = THREE.MathUtils.lerp(
          p.position.x,
          (i === 0 ? 1 : -1) * 0.064 + (flying ? 0 : gaze.x * 0.032),
          blend,
        );
        p.position.y = THREE.MathUtils.lerp(
          p.position.y,
          -0.088 + (flying ? 0 : gaze.y * 0.035),
          blend,
        );
      });
      if (time < waveUntil) {
        const envelope = Math.min(1, (waveUntil - time) * 3);
        arms[0].rotation.z -= envelope * (0.72 + Math.sin(time * 12) * 0.18);
        arms[0].rotation.x -= envelope * 0.16;
      }
    },
  };
}
