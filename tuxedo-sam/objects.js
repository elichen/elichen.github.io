const B = window.BABYLON;
export const V = (x = 0, y = 0, z = 0) => new B.Vector3(x, y, z);
export function atelier(scene) {
  const materials = new Map();
  function mat(color, glow = 0) {
    const key = color + glow;
    if (materials.has(key)) return materials.get(key);
    const m = new B.StandardMaterial(key, scene);
    m.diffuseColor = B.Color3.FromHexString(color);
    m.specularColor = new B.Color3(0.17, 0.21, 0.23);
    m.specularPower = 48;
    m.emissiveColor = m.diffuseColor.scale(glow);
    materials.set(key, m);
    return m;
  }
  const node = (name, parent) => {
    const n = new B.TransformNode(name, scene);
    n.parent = parent;
    return n;
  };
  function finish(m, parent, color, pos) {
    m.parent = parent;
    m.material = typeof color === "string" ? mat(color) : color;
    m.position.copyFromFloats(...pos);
    m.receiveShadows = true;
    return m;
  }
  function ball(parent, color, pos, size, segments = 20) {
    const m = B.MeshBuilder.CreateSphere(
      "sculpted detail",
      { diameter: 1, segments },
      scene,
    );
    m.scaling.copyFromFloats(...size);
    return finish(m, parent, color, pos);
  }
  function box(parent, color, pos, size) {
    const m = B.MeshBuilder.CreateBox("little detail", { size: 1 }, scene);
    m.scaling.copyFromFloats(...size);
    return finish(m, parent, color, pos);
  }
  function cone(parent, color, pos, top, bottom, height, tessellation = 24) {
    return finish(
      B.MeshBuilder.CreateCylinder(
        "turned detail",
        { diameterTop: top, diameterBottom: bottom, height, tessellation },
        scene,
      ),
      parent,
      color,
      pos,
    );
  }
  function tube(parent, color, points, radius = 0.02) {
    return finish(
      B.MeshBuilder.CreateTube(
        "curved detail",
        {
          path: points.map((p) => (Array.isArray(p) ? V(...p) : p)),
          radius,
          tessellation: 8,
        },
        scene,
      ),
      parent,
      color,
      [0, 0, 0],
    );
  }
  function ring(parent, color, pos, radius, tubeSize = 0.03) {
    return finish(
      B.MeshBuilder.CreateTorus(
        "ring",
        { diameter: radius * 2, thickness: tubeSize, tessellation: 48 },
        scene,
      ),
      parent,
      color,
      pos,
    );
  }
  return { mat, node, ball, box, cone, tube, ring };
}
export function makeSam(a) {
  const { node, ball, cone, tube } = a,
    root = node("Tuxedo Sam"),
    body = node("waddling body", root);
  const blue = "#689ed5",
    white = "#fff9e9",
    pink = "#ed8caa",
    yellow = "#f7c866",
    ink = "#243b51";
  ball(body, blue, [0, 1.02, 0], [1.74, 1.78, 1.42], 40);
  // Reference: eyes sit on blue; the white belly begins immediately below the beak.
  const belly = new B.Mesh("white belly inlay", body.getScene()),
    vd = new B.VertexData(),
    pos = [],
    uv = [],
    idx = [];
  for (let j = 0; j <= 18; j++)
    for (let i = 0; i <= 64; i++) {
      const r = j / 18,
        t = (i / 64) * Math.PI * 2,
        lat = -0.5 + Math.sin(t) * r * 0.69,
        lon = Math.cos(t) * r * 1.03;
      pos.push(
        0.877 * Math.cos(lat) * Math.sin(lon),
        1.02 + 0.896 * Math.sin(lat),
        0.717 * Math.cos(lat) * Math.cos(lon),
      );
      uv.push(i / 64, j / 18);
    }
  for (let j = 0; j < 18; j++)
    for (let i = 0; i < 64; i++) {
      const k = j * 65 + i;
      idx.push(k, k + 65, k + 1, k + 1, k + 65, k + 66);
    }
  const norms = [];
  for (let i = 0; i < pos.length; i += 3) {
    const normal = V(
      pos[i] / (0.877 * 0.877),
      (pos[i + 1] - 1.02) / (0.896 * 0.896),
      pos[i + 2] / (0.717 * 0.717),
    ).normalize();
    norms.push(normal.x, normal.y, normal.z);
  }
  vd.positions = pos;
  vd.indices = idx;
  vd.normals = norms;
  vd.uvs = uv;
  vd.applyToMesh(belly);
  belly.parent = body;
  belly.material = a.mat(white);
  belly.material.backFaceCulling = false;
  belly.receiveShadows = true;
  const eyes = [];
  for (const s of [-1, 1]) {
    eyes.push(
      ball(body, ink, [s * 0.252, 1.39, 0.64], [0.092, 0.145, 0.056], 20),
    );
    ball(
      body,
      white,
      [s * 0.252 - 0.013, 1.425, 0.669],
      [0.025, 0.032, 0.013],
      12,
    );
    ball(body, "#f2b1b7", [s * 0.425, 1.24, 0.63], [0.2, 0.088, 0.025], 16);
  }
  ball(body, yellow, [0, 1.192, 0.727], [0.36, 0.19, 0.27], 24);
  tube(
    body,
    "#d49a42",
    [
      [-0.12, 1.173, 0.835],
      [0, 1.153, 0.862],
      [0.12, 1.173, 0.835],
    ],
    0.009,
  );
  const bow = node("favorite bow tie", body);
  bow.position = V(0, 0.985, 0.764);
  for (const s of [-1, 1]) {
    const b = ball(bow, pink, [s * 0.145, 0, 0], [0.31, 0.25, 0.11]);
    b.rotation.z = s * -0.3;
    tube(
      bow,
      "#cc6c8d",
      [
        [s * 0.06, 0, 0.057],
        [s * 0.2, 0.045, 0.056],
      ],
      0.008,
    );
  }
  ball(bow, "#f3a4bb", [0, 0, 0.032], [0.145, 0.16, 0.12]);
  const hat = node("jaunty sailor hat", body);
  hat.position = V(0.3, 1.82, -0.03);
  hat.rotation.z = -0.25;
  cone(hat, white, [0, 0.035, 0], 0.85, 0.85, 0.105);
  cone(hat, white, [0, 0.155, 0], 0.52, 0.65, 0.15);
  cone(hat, pink, [0, 0.08, 0], 0.66, 0.66, 0.075);
  ball(hat, white, [0, 0.231, 0], [0.54, 0.16, 0.54]);
  ball(hat, pink, [0.32, 0.09, 0.025], [0.14, 0.13, 0.11]);
  for (const z of [-1, 1]) {
    const tail = ball(hat, pink, [0.4, 0.07, z * 0.055], [0.18, 0.085, 0.1]);
    tail.rotation.y = z * 0.4;
  }
  const arms = [];
  for (const s of [-1, 1]) {
    const arm = node("flipper", body);
    arm.position = V(s * 0.77, 1.13, -0.015);
    const m = ball(arm, blue, [s * 0.11, -0.24, 0], [0.3, 0.7, 0.36]);
    m.rotation.z = s * 0.25;
    arms.push(arm);
  }
  const feet = [];
  for (const s of [-1, 1]) {
    const foot = node("little yellow shoe", root);
    foot.position = V(s * 0.36, 0.16, 0.13);
    ball(foot, yellow, [0, 0, 0.1], [0.45, 0.24, 0.66]);
    tube(
      foot,
      "#e6ad4c",
      [
        [s * 0.05, 0.05, 0.4],
        [s * 0.05, 0.075, 0.32],
      ],
      0.01,
    );
    feet.push(foot);
  }
  // A tiny travel satchel and stitched strap around the back.
  tube(
    body,
    "#d2ac79",
    [
      [-0.58, 1.47, -0.32],
      [-0.29, 1.2, -0.64],
      [0.2, 0.66, -0.62],
      [0.56, 0.44, -0.29],
    ],
    0.045,
  );
  const bag = node("overnight satchel", body);
  bag.position = V(0.64, 0.56, -0.36);
  bag.rotation.z = 0.12;
  ball(bag, "#b98761", [0, 0, 0], [0.48, 0.53, 0.3]);
  boxDetail();
  function boxDetail() {
    a.box(bag, "#d0a175", [0, 0.105, -0.13], [0.43, 0.15, 0.05]);
    a.box(bag, yellow, [0, 0.02, -0.167], [0.09, 0.1, 0.025]);
  }
  return {
    root,
    body,
    eyes,
    arms,
    feet,
    hat,
    bow,
    update(t, walking, wave) {
      const phase = t * 7.3,
        amount = walking ? 1 : 0;
      body.position.y = Math.abs(Math.sin(phase)) * 0.045 * amount;
      body.rotation.z = Math.sin(phase) * 0.06 * amount;
      body.rotation.x = 0.045 * amount;
      feet.forEach((f, i) => {
        const p = phase + i * Math.PI;
        f.position.y = 0.16 + Math.max(0, Math.sin(p)) * 0.14 * amount;
        f.position.z = 0.13 + Math.cos(p) * 0.2 * amount;
        f.rotation.x = Math.sin(p) * 0.2 * amount;
      });
      arms.forEach((arm, i) => {
        arm.rotation.x = Math.cos(phase + i * Math.PI) * 0.25 * amount;
        arm.rotation.z =
          i === 1 && wave > 0
            ? -0.95 - Math.sin(wave * 15) * 0.25
            : Math.sin(phase + i * Math.PI) * 0.08 * amount;
      });
      const blink = t % 4.7;
      eyes.forEach((e) => (e.scaling.y = blink > 4.52 ? 0.023 : 0.145));
      hat.rotation.z = -0.25 + Math.sin(phase) * 0.018 * amount;
      bow.rotation.z = Math.sin(phase) * 0.025 * amount;
    },
  };
}
export function makeLandmark(a, type, parent) {
  const { node, ball, box, cone, tube, ring } = a;
  const root = node(type, parent);
  const cream = "#f5e7c8",
    gold = "#d9b478",
    pink = "#df9e96",
    dark = "#537b7a";
  if (type === "home") {
    ball(root, "#f3f5df", [0, 0.04, 0], [1.15, 0.15, 0.8]);
    ball(root, "#e4f0e7", [0, 0.22, 0], [0.55, 0.45, 0.55]);
    ball(root, "#8bbdc0", [0, 0.12, 0.263], [0.2, 0.26, 0.07]);
    for (let i = 0; i < 5; i++) {
      const theta = (i * Math.PI) / 4;
      tube(
        root,
        "#cadfd8",
        [
          [Math.cos(theta) * 0.265, 0.2, Math.sin(theta) * 0.26],
          [Math.cos(theta) * 0.19, 0.36, Math.sin(theta) * 0.19],
        ],
        0.008,
      );
    }
    const flag = box(root, "#eb9aa9", [0.48, 0.48, 0], [0.25, 0.15, 0.02]);
    tube(
      root,
      cream,
      [
        [0.35, 0, 0],
        [0.35, 0.62, 0],
      ],
      0.014,
    );
    ball(root, "#e8eedf", [-0.4, 0.13, 0.12], [0.22, 0.22, 0.22]);
    ball(root, dark, [-0.4, 0.15, 0.224], [0.035, 0.04, 0.02]);
  } else if (type === "cape") {
    cone(root, "#83a89a", [0, 0.2, 0], 0.6, 0.95, 0.4, 5);
    cone(root, "#b7c4a0", [0, 0.4, 0], 0.65, 0.67, 0.06, 5);
    for (let i = 0; i < 4; i++) {
      const x = (i - 1.5) * 0.19;
      box(
        root,
        [cream, pink, "#eec788", "#9fc5b5"][i],
        [x, 0.11, 0.45],
        [0.15, 0.23, 0.15],
      );
      cone(root, "#b47768", [x, 0.25, 0.45], 0, 0.23, 0.1, 4);
    }
  } else if (type === "paris") {
    for (const s of [-1, 1])
      for (const z of [-1, 1])
        tube(
          root,
          gold,
          [
            [s * 0.22, 0, z * 0.22],
            [s * 0.105, 0.4, z * 0.105],
            [s * 0.055, 0.73, z * 0.055],
            [0, 1.1, 0],
          ],
          0.035,
        );
    box(root, gold, [0, 0.35, 0], [0.39, 0.045, 0.39]);
    box(root, cream, [0, 0.66, 0], [0.22, 0.04, 0.22]);
    tube(
      root,
      gold,
      [
        [0, 1, 0],
        [0, 1.25, 0],
      ],
      0.014,
    );
    for (let i = 0; i < 3; i++)
      for (const z of [-1, 1])
        tube(
          root,
          gold,
          [
            [-0.18 + i * 0.05, 0.13 + i * 0.18, z * (0.18 - i * 0.045)],
            [0.13 - i * 0.025, 0.31 + i * 0.16, z * (0.13 - i * 0.035)],
          ],
          0.012,
        );
  } else if (type === "london") {
    box(root, cream, [0, 0.39, 0], [0.25, 0.78, 0.25]);
    box(root, gold, [0, 0.7, 0], [0.32, 0.22, 0.32]);
    cone(root, dark, [0, 0.92, 0], 0, 0.38, 0.3, 4);
    tube(
      root,
      gold,
      [
        [0, 1.02, 0],
        [0, 1.16, 0],
      ],
      0.013,
    );
    for (const z of [-1, 1]) {
      ball(root, "#fff5d8", [0, 0.73, z * 0.165], [0.16, 0.16, 0.025]);
      tube(
        root,
        dark,
        [
          [0, 0.78, z * 0.181],
          [0, 0.73, z * 0.181],
          [0.045, 0.71, z * 0.181],
        ],
        0.008,
      );
    }
    for (let i = 0; i < 4; i++)
      box(root, dark, [0, 0.15 + i * 0.115, 0.13], [0.05, 0.065, 0.01]);
    box(root, "#cf817b", [0.3, 0.12, 0.03], [0.15, 0.24, 0.14]);
  } else if (type === "arctic") {
    for (let i = 0; i < 5; i++) {
      cone(
        root,
        i % 2 ? "#e9f3e3" : "#bbd7cc",
        [(i - 2) * 0.2, 0.14, Math.sin(i) * 0.13],
        0,
        0.35,
        0.3 + (i % 2) * 0.25,
        5,
      );
    }
    ball(root, cream, [0.2, 0.16, 0.3], [0.36, 0.2, 0.17]);
    ball(root, cream, [0.38, 0.21, 0.3], [0.15, 0.15, 0.15]);
    for (const s of [-1, 1])
      ball(root, cream, [0.37, 0.28, 0.3 + s * 0.06], [0.07, 0.07, 0.07]);
    ball(root, dark, [0.451, 0.2, 0.3], [0.03, 0.04, 0.05]);
  } else if (type === "japan") {
    cone(root, "#95b3ac", [0.4, 0.21, -0.16], 0, 0.78, 0.55, 7);
    cone(root, "#f4efdc", [0.4, 0.42, -0.16], 0, 0.31, 0.23, 7);
    for (let k = 0; k < 3; k++) {
      box(
        root,
        cream,
        [-0.17, 0.14 + k * 0.2, 0.16],
        [0.32 - k * 0.055, 0.17, 0.26 - k * 0.04],
      );
      cone(
        root,
        "#b8766b",
        [-0.17, 0.245 + k * 0.2, 0.16],
        0.18 - k * 0.04,
        0.52 - k * 0.08,
        0.09,
        4,
      );
    }
    for (const s of [-1, 1]) {
      tube(
        root,
        "#aa7b61",
        [
          [s * 0.48, 0, 0.35],
          [s * 0.48, 0.32, 0.35],
        ],
        0.025,
      );
      for (let j = 0; j < 4; j++)
        ball(
          root,
          "#efb4bb",
          [
            s * 0.48 + Math.sin(j * 2) * 0.1,
            0.33 + j * 0.025,
            0.35 + Math.cos(j * 2) * 0.1,
          ],
          [0.23, 0.2, 0.22],
        );
    }
  } else if (type === "sydney") {
    ball(root, "#b7c9a7", [0, 0.02, 0], [0.85, 0.07, 0.5]);
    for (let i = 0; i < 4; i++) {
      const sail = ball(
        root,
        cream,
        [(i - 1.5) * 0.15, 0.15 + Math.sin(i) * 0.04, 0],
        [0.19, 0.43, 0.3],
      );
      sail.rotation.z = -0.3 - i * 0.12;
    }
    tube(
      root,
      dark,
      [
        [-0.5, 0, -0.2],
        [-0.3, 0.3, -0.2],
        [0.2, 0.36, -0.2],
        [0.5, 0, -0.2],
      ],
      0.026,
    );
  }
  return root;
}
export function makeTree(a, parent, kind = "pine", scale = 1) {
  const n = a.node("miniature tree", parent);
  n.scaling.setAll(scale);
  a.cone(n, "#997d5e", [0, 0.13, 0], 0.035, 0.05, 0.26, 6);
  if (kind === "pine") {
    a.cone(n, "#629e91", [0, 0.29, 0], 0, 0.24, 0.35, 7);
    a.cone(n, "#8fb7a0", [0, 0.42, 0], 0, 0.19, 0.28, 7);
  } else {
    a.ball(n, "#a6c7a1", [0, 0.32, 0], [0.32, 0.37, 0.3], 12);
    a.ball(n, "#c1d2a5", [0.08, 0.4, 0], [0.23, 0.24, 0.22], 12);
  }
  return n;
}
export function makePlane(a, parent) {
  const n = a.node("little airmail plane", parent);
  a.ball(n, "#efd9b0", [0, 0, 0], [0.18, 0.19, 0.65]);
  a.ball(n, "#df9c88", [0, 0.015, 0.24], [0.18, 0.17, 0.14]);
  a.ball(n, "#5a8185", [0, 0.09, 0.04], [0.11, 0.07, 0.17]);
  a.box(n, "#eed8b5", [0, -0.01, 0], [0.84, 0.045, 0.19]);
  a.box(n, "#df9c88", [0, 0.02, -0.25], [0.34, 0.03, 0.1]);
  a.box(n, "#df9c88", [0, 0.1, -0.25], [0.025, 0.21, 0.12]);
  const prop = a.box(n, "#526f77", [0, 0, 0.34], [0.37, 0.028, 0.025]);
  return { root: n, prop };
}
