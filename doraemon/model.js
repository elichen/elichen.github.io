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
  const porcelain = (color, roughness, clearcoat) =>
    new THREE.MeshPhysicalMaterial({
      color,
      roughness,
      metalness: 0,
      clearcoat,
      clearcoatRoughness: 0.32,
    });
  const blue = porcelain("#009edc", 0.4, 0.28);
  const white = porcelain("#fffdf3", 0.46, 0.13);
  const ink = material("#182b32", { roughness: 0.8 });
  const red = porcelain("#ee303d", 0.26, 0.42);
  const gold = material("#f2c13d", { metalness: 0.52, roughness: 0.25 });
  const bamboo = material("#f1cb76", { roughness: 0.46 });
  const bambooEdge = material("#bd933f", { roughness: 0.5 });
  const mouthMat = material("#ffffff", { roughness: 0.94, vertexColors: true });
  const tongueMat = porcelain("#ef8373", 0.62, 0.08);

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
  const skull = ball(head, blue, 0, 1.99, 0, 1.63, 1.55, 1.43);

  const faceZ = (x, y, lift = 0.03) =>
    1.43 *
      Math.sqrt(
        Math.max(0.012, 1 - (x / 1.63) ** 2 - ((y - 1.99) / 1.55) ** 2),
      ) +
    lift;
  // A real aperture and a gently recessed interior give the smile volume.
  // The lower curve is a broad bowl, like the 3D flight reference, rather
  // than a pointed wedge. The white face stays one continuous surface.
  const mouthHalf = 1.025;
  const mouthTop = (x) =>
    1.84 - 0.052 * Math.cos((x / mouthHalf) * Math.PI * 0.5);
  const mouthBottom = (x) =>
    1.84 -
    0.81 * Math.pow(Math.max(0, Math.cos((x / mouthHalf) * Math.PI * 0.5)), 0.67);
  const maskShape = new THREE.Shape();
  maskShape.absellipse(0, 1.76, 1.415, 1.245, 0, Math.PI * 2, false, 0);
  const mouthHole = new THREE.Path();
  mouthHole.moveTo(-mouthHalf, mouthTop(-mouthHalf));
  for (let i = 1; i <= 64; i++) {
    const x = -mouthHalf + (i / 64) * mouthHalf * 2;
    mouthHole.lineTo(x, mouthTop(x));
  }
  for (let i = 63; i >= 0; i--) {
    const x = -mouthHalf + (i / 64) * mouthHalf * 2;
    mouthHole.lineTo(x, mouthBottom(x));
  }
  mouthHole.closePath();
  maskShape.holes.push(mouthHole);

  // Subdivide in the face plane before projecting, so the aperture and
  // silhouette remain smooth even in a close three-quarter camera view.
  const flatMask = new THREE.ShapeGeometry(maskShape, 48);
  const maskPositions = flatMask.getAttribute("position");
  const maskVertices = [];
  const projectTriangle = (a, b, c, depth = 0) => {
    const ab = a.distanceToSquared(b),
      bc = b.distanceToSquared(c),
      ca = c.distanceToSquared(a);
    if (depth < 11 && Math.max(ab, bc, ca) > 0.01) {
      if (ab >= bc && ab >= ca) {
        const mid = a.clone().add(b).multiplyScalar(0.5);
        projectTriangle(a, mid, c, depth + 1);
        projectTriangle(mid, b, c, depth + 1);
      } else if (bc >= ca) {
        const mid = b.clone().add(c).multiplyScalar(0.5);
        projectTriangle(a, b, mid, depth + 1);
        projectTriangle(a, mid, c, depth + 1);
      } else {
        const mid = c.clone().add(a).multiplyScalar(0.5);
        projectTriangle(a, b, mid, depth + 1);
        projectTriangle(mid, b, c, depth + 1);
      }
      return;
    }
    for (const p of [a, b, c])
      maskVertices.push(p.x, p.y, faceZ(p.x, p.y, 0.036));
  };
  for (let i = 0; i < flatMask.index.count; i += 3) {
    const points = [0, 1, 2].map((j) =>
      new THREE.Vector3().fromBufferAttribute(maskPositions, flatMask.index.getX(i + j)),
    );
    projectTriangle(...points);
  }
  flatMask.dispose();
  const maskGeo = new THREE.BufferGeometry();
  maskGeo.setAttribute("position", new THREE.Float32BufferAttribute(maskVertices, 3));
  const maskNormals = [];
  for (let i = 0; i < maskVertices.length; i += 3) {
    const normal = new THREE.Vector3(
      maskVertices[i] / 1.63 ** 2,
      (maskVertices[i + 1] - 1.99) / 1.55 ** 2,
      (maskVertices[i + 2] - 0.036) / 1.43 ** 2,
    ).normalize();
    maskNormals.push(normal.x, normal.y, normal.z);
  }
  maskGeo.setAttribute("normal", new THREE.Float32BufferAttribute(maskNormals, 3));
  const faceMask = new THREE.Mesh(maskGeo, white);
  faceMask.receiveShadow = true;
  head.add(faceMask);

  // Carve the hidden blue shell behind the mouth. Its generous margin is
  // covered by the white mask; the mouth surface closes the actual opening.
  skull.geometry = skull.geometry.clone();
  const skullPoints = skull.geometry.getAttribute("position");
  const skullIndices = [];
  for (let i = 0; i < skull.geometry.index.count; i += 3) {
    const ids = [0, 1, 2].map((j) => skull.geometry.index.getX(i + j));
    const p = new THREE.Vector3();
    for (const id of ids)
      p.add(new THREE.Vector3().fromBufferAttribute(skullPoints, id));
    p.multiplyScalar(1 / 3).multiply(skull.scale).add(skull.position);
    const edgeX = THREE.MathUtils.clamp(p.x, -mouthHalf, mouthHalf);
    const inMouth = p.z > 0 && Math.abs(p.x) < mouthHalf + 0.17 &&
      p.y < mouthTop(edgeX) + 0.16 && p.y > mouthBottom(edgeX) - 0.17;
    if (!inMouth) skullIndices.push(...ids);
  }
  skull.geometry.setIndex(skullIndices);

  const vertices = [],
    colors = [],
    indices = [],
    columns = 64,
    rows = 22;
  const darkMouth = new THREE.Color("#4d151e");
  const warmMouth = new THREE.Color("#a3423c");
  for (let j = 0; j <= rows; j++)
    for (let i = 0; i <= columns; i++) {
      const t = i / columns,
        x = (t - 0.5) * mouthHalf * 2;
      const top = mouthTop(x);
      const bottom = mouthBottom(x);
      const y = THREE.MathUtils.lerp(top, bottom, j / rows);
      const depth = Math.sin(t * Math.PI) * Math.sin((j / rows) * Math.PI) * 0.18;
      vertices.push(x, y, faceZ(x, y, 0.03 - depth));
      const color = darkMouth.clone().lerp(warmMouth, Math.pow(j / rows, 1.5));
      colors.push(color.r, color.g, color.b);
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
  smileGeo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  smileGeo.setIndex(indices);
  smileGeo.computeVertexNormals();
  head.add(new THREE.Mesh(smileGeo, mouthMat));

  // The tongue ends on the lower smile curve; it cannot spill onto the chin.
  const tonguePositions = [], tongueIndices = [];
  const tongueHalf = 0.52;
  for (let j = 0; j <= 10; j++)
    for (let i = 0; i <= 40; i++) {
      const x = (i / 40 - 0.5) * tongueHalf * 2;
      const bottom = mouthBottom(x) + 0.018;
      const top = mouthBottom(tongueHalf) + 0.018 +
        Math.cos((x / tongueHalf) * Math.PI * 0.5) * 0.12;
      const y = THREE.MathUtils.lerp(top, bottom, j / 10);
      const bulge = Math.sin((i / 40) * Math.PI) * Math.sin((j / 10) * Math.PI) * 0.042;
      tonguePositions.push(x, y, faceZ(x, y, 0.052 + bulge));
    }
  for (let j = 0; j < 10; j++)
    for (let i = 0; i < 40; i++) {
      const a = j * 41 + i;
      tongueIndices.push(a, a + 41, a + 1, a + 1, a + 41, a + 42);
    }
  const tongueGeo = new THREE.BufferGeometry();
  tongueGeo.setAttribute("position", new THREE.Float32BufferAttribute(tonguePositions, 3));
  tongueGeo.setIndex(tongueIndices);
  tongueGeo.computeVertexNormals();
  head.add(new THREE.Mesh(tongueGeo, tongueMat));
  for (const lower of [false, true]) {
    const points = [];
    for (let i = 0; i <= 64; i++) {
      const x = (i / 64 - 0.5) * mouthHalf * 2;
      const y = lower ? mouthBottom(x) : mouthTop(x);
      points.push([x, y, faceZ(x, y, 0.045)]);
    }
    tube(head, points, lower ? 0.025 : 0.037, white);
  }
  tube(
    head,
    [
      [0, 2.53, faceZ(0, 2.53, 0.054)],
      [0, 2.15, faceZ(0, 2.15, 0.054)],
      [0, mouthTop(0) + 0.026, faceZ(0, mouthTop(0), 0.065)],
    ],
    0.018,
    ink,
  );

  // Neighboring oval eyes follow the head's curvature. Morphing the caps
  // along that same surface keeps a blink from collapsing into floating discs.
  const eyes = [],
    pupils = [];
  const eyeZ = (x, y, cx, cy, lift = 0.05) => {
    const radial = ((x - cx) / 0.333) ** 2 + ((y - cy) / 0.395) ** 2;
    return faceZ(x, y, lift + Math.max(0, 1 - radial) * 0.068);
  };
  for (const side of [-1, 1]) {
    const eye = new THREE.Group();
    const cx = side * 0.325, cy = 2.91;
    eye.position.set(cx, cy, 0);
    head.add(eye);
    const eyeSurfaces = [];
    for (const outline of [true, false]) {
      const eyePositions = [], closedPositions = [], eyeIndices = [];
      const rx = outline ? 0.337 : 0.330;
      const ry = outline ? 0.398 : 0.391;
      const lift = outline ? 0.047 : 0.054;
      for (let j = 0; j <= 12; j++)
        for (let i = 0; i <= 64; i++) {
          const angle = (i / 64) * Math.PI * 2, r = j / 12;
          const x = Math.cos(angle) * rx * r;
          const y = Math.sin(angle) * ry * r;
          eyePositions.push(x, y, eyeZ(cx + x, cy + y, cx, cy, lift));
          closedPositions.push(x, y * 0.055,
            eyeZ(cx + x, cy + y * 0.055, cx, cy, lift));
        }
      for (let j = 0; j < 12; j++)
        for (let i = 0; i < 64; i++) {
          const a = j * 65 + i;
          eyeIndices.push(a, a + 65, a + 1, a + 1, a + 65, a + 66);
        }
      const eyeGeo = new THREE.BufferGeometry();
      eyeGeo.setAttribute("position", new THREE.Float32BufferAttribute(eyePositions, 3));
      eyeGeo.setIndex(eyeIndices);
      eyeGeo.computeVertexNormals();
      eyeGeo.morphAttributes.position = [new THREE.Float32BufferAttribute(closedPositions, 3)];
      const surface = new THREE.Mesh(eyeGeo, outline ? ink : white);
      eye.add(surface);
      eyeSurfaces.push(surface);
    }
    const pupil = ball(
      eye,
      ink,
      -side * 0.064,
      -0.12,
      eyeZ(cx - side * 0.064, cy - 0.12, cx, cy) + 0.009,
      0.074,
      0.105,
      0.024,
    );
    pupil.userData.eyeCenter = { x: cx, y: cy };
    pupils.push(pupil);
    ball(pupil, white, -0.2, 0.32, 0.83, 0.23, 0.19, 0.22);
    const lidPoints = [];
    for (let i = 0; i <= 24; i++) {
      const x = (i / 24 - 0.5) * 0.52;
      const y = -0.065 + Math.sin((i / 24) * Math.PI) * 0.075;
      lidPoints.push([x, y, eyeZ(cx + x, cy + y, cx, cy, 0.078)]);
    }
    const closedLid = tube(eye, lidPoints, 0.018, ink);
    closedLid.visible = false;
    eyes.push({ surfaces: eyeSurfaces, closedLid, pupil });
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
        const y = 2.27 + (i - 1) * (0.15 + t * 0.15);
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
  const tail = new THREE.Group();
  tail.position.set(0, -0.38, -0.86);
  torso.add(tail);
  tube(
    tail,
    [
      [0, 0, 0],
      [0, 0.01, -0.26],
      [0, 0.1, -0.36],
    ],
    0.05,
    red,
  );
  const tailAnchor = ball(tail, red, 0, 0.13, -0.39, 0.205);

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
    winkAt = -10,
    waveStarted = -10,
    waveUntil = 0,
    posePitch = 0,
    previousSpeed = 0,
    bellPitch = 0,
    bellVelocity = 0;
  const pupilNormal = new THREE.Vector3();
  const pupilFront = new THREE.Vector3(0, 0, 1);
  const wink = () => {
    winkAt = elapsed + 0.01;
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
      waveStarted = elapsed;
      waveUntil = elapsed + 2.6;
      winkAt = elapsed + 0.28;
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
      const lookX = THREE.MathUtils.clamp(gaze.x || 0, -1, 1);
      const lookY = THREE.MathUtils.clamp(gaze.y || 0, -1, 1);
      const greeting = THREE.MathUtils.smoothstep(time - waveStarted, 0, 0.28) *
        THREE.MathUtils.smoothstep(waveUntil - time, 0, 0.5);
      posePitch = THREE.MathUtils.lerp(posePitch, glide * 0.9, blend);
      flightRig.rotation.x = posePitch;
      headPivot.rotation.x = -posePitch * 0.98;
      headPivot.rotation.y = THREE.MathUtils.lerp(headPivot.rotation.y, lookX * 0.07, blend);
      headPivot.rotation.z = reduced ? 0 :
        Math.sin(time * 0.73) * 0.015 + greeting * 0.045;
      torso.scale.x = torso.scale.z = reduced ? 1 : 1 + Math.sin(time * 1.35) * 0.003;
      rotor.rotation.y += dt * (reduced ? 9 : 24 + speed * 1.4);
      blurMat.opacity = flying ? 0.08 : 0.035;
      ghostMaterial.opacity = flying ? 0.11 : 0.065;
      const acceleration = dt > 0 ? (speed - previousSpeed) / dt : 0;
      previousSpeed = speed;
      const bellTarget = reduced ? 0 : THREE.MathUtils.clamp(acceleration * -0.018, -0.2, 0.2) +
        Math.sin(time * 2.2) * 0.045;
      bellVelocity += ((bellTarget - bellPitch) * 32 - bellVelocity * 7) * dt;
      bellPitch += bellVelocity * dt;
      bell.rotation.x = bellPitch;
      bell.rotation.z = reduced ? 0 : -root.rotation.z * 0.35;
      tail.rotation.y = reduced ? 0 : Math.sin(time * 1.7) * (0.09 + glide * 0.04);
      tail.rotation.x = reduced ? 0 : Math.sin(time * 2.3) * 0.06;
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
      if (time > blinkAt + 0.22) blinkAt = time + 3.3 + Math.random() * 3;
      const blinkAmount = (start, duration) => {
        const phase = (time - start) / duration;
        return phase >= 0 && phase <= 1 ? Math.sin(phase * Math.PI) ** 1.2 : 0;
      };
      const blink = blinkAmount(blinkAt, 0.22);
      const winking = blinkAmount(winkAt, 0.42);
      eyes.forEach((eye, i) => {
        const closure = Math.max(blink, i === 1 ? winking : 0);
        eye.surfaces.forEach((surface) => {
          surface.morphTargetInfluences[0] = closure * 0.16;
        });
        eye.closedLid.visible = closure >= 0.68;
        eye.pupil.visible = closure < 0.68;
        eye.pupil.scale.y = 0.105 * (1 - closure * 0.85);
      });
      pupils.forEach((p, i) => {
        p.position.x = THREE.MathUtils.lerp(
          p.position.x,
          (i === 0 ? 1 : -1) * 0.064 + lookX * 0.085,
          blend,
        );
        p.position.y = THREE.MathUtils.lerp(
          p.position.y,
          -0.12 + lookY * 0.075,
          blend,
        );
        const center = p.userData.eyeCenter;
        const px = center.x + p.position.x, py = center.y + p.position.y;
        p.position.z = eyeZ(px, py, center.x, center.y) + 0.009;
        pupilNormal.set(px / 1.63 ** 2, (py - 1.99) / 1.55 ** 2,
          faceZ(px, py, 0) / 1.43 ** 2).normalize();
        p.quaternion.setFromUnitVectors(pupilFront, pupilNormal);
      });
      if (time < waveUntil) {
        arms[0].rotation.z -= greeting * (0.78 + (reduced ? 0 : Math.sin(time * 11) * 0.17));
        arms[0].rotation.x -= greeting * 0.16;
      }
    },
  };
}
