import { V, makeLandmark, makeTree, makePlane } from "./objects.js";
const B = window.BABYLON,
  TAU = Math.PI * 2,
  R = 3.15;
export const stops = [
  {
    name: "Tuxedo Island",
    region: "WHERE IT ALL BEGINS",
    lat: -76,
    lon: 0,
    type: "home",
    symbol: "❄",
    note: "A little wave goodbye. An extra bow tie packed. The whole world ahead.",
  },
  {
    name: "Cape Town",
    region: "A LITTLE SOUTHERN SUNSHINE",
    lat: -34,
    lon: 18,
    type: "cape",
    symbol: "☀",
    note: "A mountain with a tabletop. The perfect place to unpack a very small picnic.",
  },
  {
    name: "Paris",
    region: "ONE CROISSANT, S’IL VOUS PLAÎT",
    lat: 49,
    lon: 2,
    type: "paris",
    symbol: "♜",
    note: "A tower that tickles the clouds. Sam wonders if they serve ice cream up there.",
  },
  {
    name: "London",
    region: "TEA TIME, ANY TIME",
    lat: 51.5,
    lon: -0.1,
    type: "london",
    symbol: "◷",
    note: "A familiar little drizzle. Time for tea, two biscuits, and a very smart bow tie.",
  },
  {
    name: "The Arctic",
    region: "A LETTER FROM THE NORTH",
    lat: 79,
    lon: 25,
    type: "arctic",
    symbol: "✧",
    note: "Everything is quiet and sugar-white. A polar bear waves from the next iceberg.",
  },
  {
    name: "Tokyo",
    region: "BENEATH THE CHERRY BLOSSOMS",
    lat: 36,
    lon: 140,
    type: "japan",
    symbol: "✿",
    note: "Pink petals on the breeze. One lands on his hat. He decides to keep it.",
  },
  {
    name: "Sydney",
    region: "THE LONG, LOVELY WAY HOME",
    lat: -34,
    lon: 151,
    type: "sydney",
    symbol: "≈",
    note: "Sails in the harbor, salt in the air. Just one more look before heading home.",
  },
];
export function point(lat, lon, r = R) {
  lat *= Math.PI / 180;
  lon *= Math.PI / 180;
  return V(
    Math.cos(lat) * Math.sin(lon) * r,
    Math.sin(lat) * r,
    Math.cos(lat) * Math.cos(lon) * r,
  );
}
export async function makeWorld(scene, a) {
  const { node, ball, cone, tube, mat } = a,
    root = node("a world of small wonders");
  const response = await fetch("./land.geojson");
  if (!response.ok)
    throw Error("The little atlas could not be loaded. Please reload.");
  const geo = await response.json();
  const tex = new B.DynamicTexture(
    "hand-painted atlas",
    { width: 2048, height: 1024 },
    scene,
    false,
  );
  const ctx = tex.getContext(),
    W = 2048,
    H = 1024;
  ctx.fillStyle = "#549ca8";
  ctx.fillRect(0, 0, W, H);
  const gradient = ctx.createLinearGradient(0, 0, 0, H);
  gradient.addColorStop(0, "#b6d8d1");
  gradient.addColorStop(0.23, "#6aafb7");
  gradient.addColorStop(0.5, "#579fae");
  gradient.addColorStop(0.8, "#7cbac0");
  gradient.addColorStop(1, "#d0e4d8");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, W, H);
  function path(rings) {
    ctx.beginPath();
    for (const ring of rings) {
      ring.forEach(([lon, lat], i) => {
        const x = ((lon + 180) / 360) * W,
          y = ((90 - lat) / 180) * H;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath();
    }
  }
  const polygons = [];
  for (const f of geo.features) {
    if (f.geometry.type === "Polygon") polygons.push(f.geometry.coordinates);
    else if (f.geometry.type === "MultiPolygon")
      polygons.push(...f.geometry.coordinates);
  }
  // Layered coast strokes give the printed atlas shallow shelves and a warm shoreline.
  for (const width of [15, 8, 3]) {
    ctx.strokeStyle =
      width === 15 ? "#77b4b9" : width === 8 ? "#93c6c1" : "#d1d5af";
    ctx.lineWidth = width;
    for (const poly of polygons) {
      path(poly);
      ctx.stroke();
    }
  }
  ctx.fillStyle = "#c7d6ac";
  for (const poly of polygons) {
    path(poly);
    ctx.fill("evenodd");
  }
  const landPixels = ctx.getImageData(0, 0, W, H).data;
  function isLand(lat, lon) {
    const x = Math.min(W - 1, Math.max(0, Math.floor(((lon + 180) / 360) * W))),
      y = Math.min(H - 1, Math.max(0, Math.floor(((90 - lat) / 180) * H)));
    const i = (y * W + x) * 4;
    return landPixels[i] === 199 && landPixels[i + 1] === 214;
  }
  let seed = 529;
  const rnd = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  // Fine flecks and survey lines make the world feel like a painted keepsake.
  for (let i = 0; i < 14000; i++) {
    const x = rnd() * W,
      y = rnd() * H;
    ctx.fillStyle = i % 2 ? "#fff7d50b" : "#173e4810";
    ctx.fillRect(x, y, 1 + rnd() * 3, 0.7 + rnd() * 1.5);
  }
  ctx.strokeStyle = "#e3f1d529";
  ctx.lineWidth = 0.7;
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = ((90 - lat) / 180) * H;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }
  for (let lon = -180; lon < 180; lon += 30) {
    const x = ((lon + 180) / 360) * W;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  tex.update(false);
  const globe = new B.Mesh("the painted Earth", scene),
    positions = [],
    normals = [],
    uvs = [],
    indices = [],
    NX = 160,
    NY = 80;
  for (let y = 0; y <= NY; y++)
    for (let x = 0; x <= NX; x++) {
      const p = point(-90 + (y / NY) * 180, -180 + (x / NX) * 360, 1);
      positions.push(p.x * R, p.y * R, p.z * R);
      normals.push(p.x, p.y, p.z);
      uvs.push(x / NX, 1 - y / NY);
    }
  for (let y = 0; y < NY; y++)
    for (let x = 0; x < NX; x++) {
      const i = y * (NX + 1) + x;
      indices.push(i, i + NX + 1, i + 1, i + 1, i + NX + 1, i + NX + 2);
    }
  const data = new B.VertexData();
  data.positions = positions;
  data.normals = normals;
  data.uvs = uvs;
  data.indices = indices;
  data.applyToMesh(globe);
  globe.parent = root;
  globe.receiveShadows = true;
  const ocean = mat("#ffffff");
  ocean.diffuseTexture = tex;
  ocean.specularColor = B.Color3.FromHexString("#7dadae").scale(0.4);
  ocean.specularPower = 90;
  globe.material = ocean;
  function anchor(lat, lon, r = R + 0.015) {
    const n = node("surface anchor", root);
    n.position = point(lat, lon, r);
    n.rotationQuaternion = B.Quaternion.FromUnitVectorsToRef(
      V(0, 1, 0),
      n.position.normalizeToNew(),
      new B.Quaternion(),
    );
    return n;
  }
  stops.forEach((s) => {
    const place = anchor(s.lat, s.lon);
    const lon = (s.lon * Math.PI) / 180;
    place.position
      .addInPlace(
        V(Math.cos(lon), 0, -Math.sin(lon)).scale(
          s.type === "london" ? -0.62 : 0.65,
        ),
      )
      .normalize()
      .scaleInPlace(R + 0.015);
    place.rotationQuaternion = B.Quaternion.FromUnitVectorsToRef(
      V(0, 1, 0),
      place.position.normalizeToNew(),
      new B.Quaternion(),
    );
    const landmark = makeLandmark(a, s.type, place);
    landmark.scaling.setAll(s.type === "paris" ? 0.65 : 0.78);
  });
  const terrain = node("all miniature terrain", root);
  for (let i = 0; i < 360; i++) {
    const lat = -57 + rnd() * 128,
      lon = -180 + rnd() * 360;
    if (!isLand(lat, lon)) continue;
    if (
      stops.some((s) => Math.abs(s.lat - lat) < 9 && Math.abs(s.lon - lon) < 10)
    )
      continue;
    const n = anchor(lat, lon);
    n.parent = terrain;
    makeTree(a, n, Math.abs(lat) > 40 ? "pine" : "round", 0.6 + rnd() * 0.65);
  }
  const ranges = [
    [
      [31, 80],
      [35, 85],
      [38, 90],
      [29, 88],
    ],
    [
      [43, 7],
      [46, 10],
      [47, 14],
    ],
    [
      [-15, -72],
      [-23, -69],
      [-32, -70],
      [-42, -72],
    ],
    [
      [43, -110],
      [50, -118],
      [58, -133],
    ],
  ];
  for (const range of ranges)
    for (const [lat, lon] of range) {
      const n = anchor(lat, lon);
      for (let j = 0; j < 3; j++) {
        const h = 0.22 + rnd() * 0.2,
          x = (j - 1) * 0.14;
        cone(n, "#90ac9d", [x, h / 2, 0], 0, 0.3, h, 5);
        cone(n, "#f1eed8", [x, h * 0.83, 0], 0, 0.115, h * 0.35, 5);
      }
    }
  for (const [lat, lon] of [
    [23, 13],
    [25, 23],
    [20, 32],
    [24, 45],
    [-24, 132],
  ]) {
    const n = anchor(lat, lon);
    for (let j = 0; j < 3; j++)
      ball(
        n,
        "#d8c99a",
        [(j - 1) * 0.14, 0.04, j * 0.03],
        [0.38, 0.13, 0.25],
        12,
      );
  }
  // Great-circle-like, smooth closed itinerary; the globe rotates under a fixed traveler.
  const routePoints = stops.map((s) => point(s.lat, s.lon, 1));
  function rawRoute(t) {
    t = ((t % 1) + 1) % 1;
    const f = t * routePoints.length,
      i = Math.floor(f),
      u = f - i;
    const p0 = routePoints[(i + routePoints.length - 1) % routePoints.length],
      p1 = routePoints[i],
      p2 = routePoints[(i + 1) % routePoints.length],
      p3 = routePoints[(i + 2) % routePoints.length];
    return B.Vector3.CatmullRom(p0, p1, p2, p3, u).normalize();
  }
  const samples = [0];
  let totalLength = 0,
    last = rawRoute(0);
  for (let i = 1; i <= 2800; i++) {
    const p = rawRoute(i / 2800);
    totalLength += B.Vector3.Distance(last, p);
    samples.push(totalLength);
    last = p;
  }
  const stopProgress = stops.map((_, i) => samples[i * 400] / totalLength);
  function route(t) {
    const distance = (((t % 1) + 1) % 1) * totalLength;
    let low = 0,
      high = 2800;
    while (low + 1 < high) {
      const mid = (low + high) >> 1;
      if (samples[mid] < distance) low = mid;
      else high = mid;
    }
    const mix = (distance - samples[low]) / (samples[high] - samples[low] || 1);
    return rawRoute((low + mix) / 2800);
  }
  for (let i = 0; i < 340; i++) {
    if (i % 3 === 2) continue;
    const p = route(i / 340).scale(R + 0.022);
    ball(root, "#f4d895", [p.x, p.y, p.z], [0.027, 0.027, 0.027], 6);
  }
  // Tiny boats out on the oceans.
  const boats = [];
  for (const [lat, lon] of [
    [10, -35],
    [-13, 61],
    [20, 170],
    [-36, -114],
    [45, -32],
  ]) {
    const n = anchor(lat, lon, R + 0.015);
    ball(n, "#eac89b", [0, 0.035, 0], [0.23, 0.1, 0.44], 12);
    tube(
      n,
      "#b49778",
      [
        [0, 0.03, 0],
        [0, 0.42, 0],
      ],
      0.013,
    );
    const sail = new B.Mesh("linen sail", scene),
      vd = new B.VertexData();
    vd.positions = [0, 0.1, -0.17, 0, 0.4, 0, 0, 0.1, 0.12];
    vd.indices = [0, 1, 2, 2, 1, 0];
    vd.normals = [1, 0, 0, 1, 0, 0, 1, 0, 0];
    vd.applyToMesh(sail);
    sail.parent = n;
    sail.material = mat("#fff0ce");
    boats.push(n);
  }
  // Merge the stationary details by material so the rich miniature stays inexpensive to draw.
  const meshes = root.getChildMeshes().filter((m) => m !== globe),
    groups = new Map();
  for (const m of meshes) {
    m.computeWorldMatrix(true);
    const key = m.material.uniqueId;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(m);
  }
  for (const group of groups.values()) {
    const merged = B.Mesh.MergeMeshes(
      group,
      true,
      true,
      undefined,
      false,
      false,
    );
    if (merged) {
      merged.parent = root;
      merged.receiveShadows = true;
      merged.name = "batched miniature scenery";
    }
  }
  const clouds = [];
  for (let i = 0; i < 15; i++) {
    const cloud = node("drifting cotton cloud");
    const lat = -65 + rnd() * 140,
      lon = -180 + rnd() * 360;
    for (let j = 0; j < 4; j++)
      ball(
        cloud,
        "#e6eee1",
        [(j - 1.5) * 0.18, Math.sin(j * 1.8) * 0.055, 0],
        [0.36, 0.18 + rnd() * 0.13, 0.26],
        12,
      );
    cloud.scaling.setAll(0.8 + rnd() * 0.6);
    clouds.push({ root: cloud, lat, lon, speed: 0.3 + rnd() * 0.6 });
  }
  const plane = makePlane(a, null);
  // A sparse, softly luminous starfield, joined into one mesh.
  const stars = [];
  for (let i = 0; i < 190; i++) {
    const p = V((rnd() - 0.5) * 55, (rnd() - 0.5) * 36, (rnd() - 0.5) * 40);
    if (p.length() < 13) p.normalize().scaleInPlace(20);
    const size = 0.015 + rnd() * 0.025;
    stars.push(
      ball(
        null,
        mat(i % 4 ? "#cad9cf" : "#eac686", 0.8),
        [p.x, p.y, p.z],
        [size, size, size],
        5,
      ),
    );
  }
  B.Mesh.MergeMeshes(stars, true, true, undefined, true, true);
  const moon = node("a pocket moon");
  moon.position = V(-4.8, 5, -5);
  ball(moon, mat("#eadfbe", 0.16), [0, 0, 0], [0.56, 0.56, 0.56], 24);
  for (const [x, y, size] of [
    [-0.11, 0.09, 0.1],
    [0.08, -0.1, 0.08],
    [0.1, 0.13, 0.055],
  ])
    ball(moon, "#c4c5af", [x, y, 0.245], [size, size, 0.018], 12);
  const twinkles = [];
  for (let i = 0; i < 13; i++) {
    const n = node("a wishing star");
    n.position = V((rnd() - 0.5) * 28, 2 + rnd() * 11, -8 - rnd() * 8);
    const material = mat("#edd9a9", 0.6);
    a.box(n, material, [0, 0, 0], [0.015, 0.13, 0.012]);
    a.box(n, material, [0, 0, 0], [0.09, 0.016, 0.012]);
    twinkles.push(n);
  }
  const orbitMat = mat("#94babc", 0.25);
  orbitMat.alpha = 0.22;
  for (let j = 0; j < 2; j++) {
    const points = [];
    for (let i = 0; i <= 180; i++) {
      const t = (i / 180) * TAU;
      points.push(
        V(
          Math.cos(t) * (5.25 + j * 0.35),
          Math.sin(t) * 0.9,
          Math.sin(t) * 4.8,
        ),
      );
    }
    const orbit = tube(null, orbitMat, points, 0.006);
    orbit.rotation.z = j ? 0.28 : -0.25;
    orbit.position.y = -0.5;
  }
  return {
    root,
    globe,
    route,
    stopProgress,
    totalLength,
    clouds,
    plane,
    R,
    update(travel, time) {
      const up = route(travel),
        forward = route(travel + 0.0001)
          .subtract(route(travel - 0.0001))
          .normalize();
      const right = B.Vector3.Cross(up, forward).normalize();
      const tangent = B.Vector3.Cross(right, up).normalize();
      root.rotationQuaternion = B.Quaternion.RotationQuaternionFromAxis(
        right,
        up,
        tangent,
      ).conjugate();
      root.computeWorldMatrix(true);
      for (const c of clouds) {
        const local = point(c.lat, c.lon + time * c.speed, R + 0.34);
        c.root.position = B.Vector3.TransformCoordinates(
          local,
          root.getWorldMatrix(),
        );
        c.root.rotationQuaternion = B.Quaternion.FromUnitVectorsToRef(
          V(0, 1, 0),
          c.root.position.normalizeToNew(),
          new B.Quaternion(),
        );
      }
      const t = time * 0.12;
      plane.root.position = V(
        Math.cos(t) * 4.75,
        Math.sin(t * 0.7) * 1.1 + 1.2,
        Math.sin(t) * 4.75,
      );
      plane.root.setDirection(
        V(-Math.sin(t), 0.16 * Math.cos(t * 0.7), Math.cos(t)),
      );
      plane.root.rotation.z = 0.16;
      plane.prop.rotation.z = time * 35;
      twinkles.forEach((star, i) =>
        star.scaling.setAll(0.65 + Math.sin(time * 0.7 + i * 1.7) * 0.3),
      );
    },
  };
}
