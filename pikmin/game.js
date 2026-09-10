import * as THREE from "./vendor/three.module.js";

const $ = (id) => document.getElementById(id);
const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
} catch (error) {
  $("loading").innerHTML =
    '<p>This little world needs WebGL.</p><p style="font:13px sans-serif">Try a browser with hardware acceleration enabled.</p>';
  throw error;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
$("world").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color("#dfe6cb");
scene.fog = new THREE.Fog("#dfe6cb", 75, 160);
const camera = new THREE.PerspectiveCamera(
  38,
  innerWidth / innerHeight,
  0.1,
  1500,
);
const ambient = new THREE.HemisphereLight("#fff7d8", "#6b8054", 1.8);
scene.add(ambient);
const sun = new THREE.DirectionalLight("#fff1ca", 2.3);
sun.position.set(-18, 32, 15);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {
  left: -35,
  right: 35,
  top: 35,
  bottom: -35,
  near: 1,
  far: 100,
});
sun.shadow.normalBias = 0.035;
sun.shadow.bias = -0.0002;
sun.shadow.radius = 3;
scene.add(sun, sun.target);
const mat = (color, extra = {}) =>
  new THREE.MeshStandardMaterial({ color, roughness: 0.78, ...extra });
const green = mat("#618247"),
  cream = mat("#f0d991"),
  skin = mat("#f6cfaa"),
  red = mat("#bd4937");
const white = mat("#fffbe9"),
  dark = mat("#29382c"),
  silver = mat("#b6bcb0", { metalness: 0.55, roughness: 0.3 });
const sphere = new THREE.SphereGeometry(1, 12, 8);
const dummy = new THREE.Object3D();
function part(
  parent,
  geometry,
  material,
  pos,
  scale = [1, 1, 1],
  rotation = [0, 0, 0],
) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...pos);
  mesh.scale.set(...scale);
  mesh.rotation.set(...rotation);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}
function ball(parent, material, pos, scale) {
  return part(parent, sphere, material, pos, scale);
}

// Soft, continuous meadow coloration in world space; no finite board or boundary.
const groundMat = mat("#94ad7c");
groundMat.onBeforeCompile = (shader) => {
  shader.vertexShader =
    "varying vec3 meadowPosition;\n" +
    shader.vertexShader.replace(
      "#include <begin_vertex>",
      "#include <begin_vertex>\nmeadowPosition = (modelMatrix * vec4(position, 1.0)).xyz;",
    );
  shader.fragmentShader =
    `varying vec3 meadowPosition;
    float hashMeadow(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float meadowNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.0-2.0*f);return mix(mix(hashMeadow(i),hashMeadow(i+vec2(1,0)),f.x),mix(hashMeadow(i+vec2(0,1)),hashMeadow(i+vec2(1,1)),f.x),f.y);}
  ` +
    shader.fragmentShader.replace(
      "#include <color_fragment>",
      `#include <color_fragment>
    float n=meadowNoise(meadowPosition.xz*.085)*.6+meadowNoise(meadowPosition.xz*.4)*.25+meadowNoise(meadowPosition.xz*3.)*.15;
    diffuseColor.rgb *= mix(vec3(.77,.86,.68),vec3(1.16,1.1,.85),n);
    float path=sin(meadowPosition.x*.055+sin(meadowPosition.z*.055)*2.4)*7.0;
    float trail=1.0-smoothstep(.0,2.2,abs(path));
    diffuseColor.rgb=mix(diffuseColor.rgb,vec3(.60,.61,.39),trail*.22);`,
    );
};
const ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// Olimar's silhouette: oversized nose, pointed ears, cream suit and glass helmet.
const captain = new THREE.Group();
scene.add(captain);
const body = new THREE.Group();
captain.add(body);
ball(body, cream, [0, 0.64, 0], [0.34, 0.42, 0.27]);
part(
  body,
  new THREE.TorusGeometry(0.29, 0.055, 8, 20),
  silver,
  [0, 0.99, 0],
  [1, 1, 1],
  [Math.PI / 2, 0, 0],
);
ball(body, skin, [0, 1.42, 0.01], [0.44, 0.4, 0.38]);
ball(body, skin, [0, 1.35, 0.41], [0.26, 0.22, 0.25]);
ball(body, skin, [-0.43, 1.39, 0], [0.14, 0.23, 0.1]);
ball(body, skin, [0.43, 1.39, 0], [0.14, 0.23, 0.1]);
ball(body, dark, [-0.18, 1.55, 0.341], [0.1, 0.021, 0.018]);
ball(body, dark, [0.18, 1.55, 0.341], [0.1, 0.021, 0.018]);
ball(body, dark, [0, 1.13, 0.32], [0.035, 0.037, 0.015]);
for (let i = -1; i <= 1; i++)
  part(
    body,
    new THREE.ConeGeometry(0.055, 0.22, 5),
    mat("#805a3f"),
    [i * 0.08, 1.84, 0],
    [1, 1, 1],
    [0, 0, -i * 0.3],
  );
const helmetMat = new THREE.MeshPhysicalMaterial({
  color: "#e9ffff",
  transparent: true,
  opacity: 0.16,
  roughness: 0.08,
  metalness: 0.1,
  depthWrite: false,
  side: THREE.FrontSide,
});
ball(body, helmetMat, [0, 1.44, 0], [0.61, 0.61, 0.58]);
const glassArc = new THREE.MeshBasicMaterial({
  color: "#ffffef",
  transparent: true,
  opacity: 0.65,
});
part(
  body,
  new THREE.TorusGeometry(0.565, 0.012, 6, 36, 1.3),
  glassArc,
  [0, 1.44, 0.08],
  [1, 1, 1],
  [0.25, 0.5, 0.1],
);
ball(body, glassArc, [-0.25, 1.74, 0.37], [0.09, 0.14, 0.015]);
part(
  body,
  new THREE.CylinderGeometry(0.018, 0.018, 0.63, 6),
  silver,
  [0.3, 2.12, -0.04],
  [1, 1, 1],
  [0, 0, -0.35],
);
const antenna = ball(
  body,
  mat("#ff6243", { emissive: "#f54a22", emissiveIntensity: 0.7 }),
  [0.41, 2.42, -0.04],
  [0.09, 0.09, 0.09],
);
part(
  body,
  new THREE.BoxGeometry(0.46, 0.52, 0.22),
  mat("#ad8470"),
  [0, 0.71, -0.3],
);
ball(body, mat("#71aeb7"), [-0.12, 0.78, 0.253], [0.045, 0.045, 0.018]);
ball(body, red, [0.1, 0.64, 0.267], [0.045, 0.045, 0.018]);
const limbs = [];
for (const side of [-1, 1]) {
  const leg = new THREE.Group();
  leg.position.set(side * 0.19, 0.36, 0);
  body.add(leg);
  limbs.push(leg);
  ball(leg, cream, [0, -0.12, 0], [0.14, 0.23, 0.14]);
  ball(leg, red, [0, -0.25, 0.075], [0.17, 0.105, 0.23]);
  const arm = new THREE.Group();
  arm.position.set(side * 0.31, 0.85, 0);
  body.add(arm);
  limbs.push(arm);
  ball(arm, cream, [side * 0.075, -0.16, 0], [0.12, 0.23, 0.12]);
  ball(arm, red, [side * 0.09, -0.34, 0.015], [0.13, 0.13, 0.12]);
}
captain.rotation.y = 0;

// Merge vertex-colored parts into one draw per Pikmin type. Capacity grows, never population-caps.
function mergeParts(parts) {
  const positions = [],
    normals = [],
    colors = [];
  for (const [geometry, color, pos, scale, rotation] of parts) {
    const g = geometry.index ? geometry.toNonIndexed() : geometry.clone();
    dummy.position.set(...pos);
    dummy.scale.set(...scale);
    dummy.rotation.set(...(rotation || [0, 0, 0]));
    dummy.updateMatrix();
    g.applyMatrix4(dummy.matrix);
    positions.push(...g.attributes.position.array);
    normals.push(...g.attributes.normal.array);
    const c = new THREE.Color(color);
    for (let j = 0; j < g.attributes.position.count; j++)
      colors.push(c.r, c.g, c.b);
    g.dispose();
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  return geometry;
}
const colors = ["#e65335", "#f6ce40", "#4989d1"];
function pikminGeometry(type, low = false) {
  const s = new THREE.SphereGeometry(1, low ? 5 : 8, low ? 3 : 5);
  const p = [];
  const add = (g, c, pos, scale, rot) => p.push([g, c, pos, scale, rot]);
  add(s, colors[type], [0, 0.34, 0], [0.115, 0.23, 0.105]);
  add(s, colors[type], [0, 0.65, 0.015], [0.18, 0.16, 0.15]);
  for (const side of [-1, 1]) {
    add(s, colors[type], [side * 0.073, 0.105, 0.035], [0.047, 0.12, 0.075]);
    if (!low)
      add(
        s,
        colors[type],
        [side * 0.16, 0.36, 0],
        [0.05, 0.16, 0.04],
        [0, 0, side * 0.6],
      );
    add(s, "#fffbea", [side * 0.074, 0.69, 0.136], [0.059, 0.065, 0.028]);
    add(s, "#263533", [side * 0.074, 0.689, 0.16], [0.025, 0.035, 0.014]);
    if (type === 1)
      add(
        s,
        colors[type],
        [side * 0.205, 0.7, 0],
        [0.14, 0.063, 0.045],
        [0, 0, side * 0.5],
      );
  }
  if (type === 0)
    add(
      new THREE.ConeGeometry(0.048, 0.18, 5),
      colors[type],
      [0, 0.64, 0.2],
      [1, 1, 1],
      [Math.PI / 2, 0, 0],
    );
  if (type === 2) add(s, "#213949", [0, 0.587, 0.14], [0.045, 0.014, 0.014]);
  add(
    new THREE.CylinderGeometry(0.015, 0.02, 0.27, 4),
    "#537b38",
    [0, 0.9, 0.01],
    [1, 1, 1],
    [0, 0, -0.18],
  );
  add(s, "#66963e", [0.11, 1.025, 0.015], [0.145, 0.025, 0.073], [0, 0, 0.34]);
  // A soft contact shadow is part of the geometry, so it costs no extra draw call.
  add(
    new THREE.CircleGeometry(0.2, 10),
    "#7c9256",
    [0, 0.012, 0],
    [1, 1, 1],
    [-Math.PI / 2, 0, 0],
  );
  const result = mergeParts(p);
  s.dispose();
  return result;
}
const pikMat = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 0.8,
});
let pikShader;
pikMat.onBeforeCompile = (shader) => {
  pikShader = shader;
  shader.uniforms.pikTime = { value: 0 };
  shader.uniforms.walking = { value: 0 };
  shader.vertexShader =
    "uniform float pikTime; uniform float walking;\n" +
    shader.vertexShader.replace(
      "#include <begin_vertex>",
      `#include <begin_vertex>
 float phase=pikTime*13.0+instanceMatrix[3].x*2.0+instanceMatrix[3].z;
 if(position.y>.025 && position.y<.22) transformed.z+=sin(phase+sign(position.x)*1.57)*.065*walking;
 if(position.y>.82) transformed.x+=sin(pikTime*2.6+instanceMatrix[3].z)*.06*(position.y-.82)*4.0;`,
    );
};
const detailed = colors.map((_, i) => pikminGeometry(i));
const simple = colors.map((_, i) => pikminGeometry(i, true));
const batches = colors.map((_, type) => ({
  type,
  capacity: 0,
  mesh: null,
  count: 0,
}));
function ensureCapacity(batch, count) {
  if (count <= batch.capacity) return;
  const capacity = Math.max(256, 2 ** Math.ceil(Math.log2(count)));
  if (batch.mesh) {
    scene.remove(batch.mesh);
    batch.mesh.dispose();
  }
  batch.mesh = new THREE.InstancedMesh(detailed[batch.type], pikMat, capacity);
  batch.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  batch.mesh.frustumCulled = false;
  batch.capacity = capacity;
  scene.add(batch.mesh);
}
let crowd = [],
  wild = [],
  elapsed = 0,
  paused = false,
  wasPaused = false,
  running = true;
const player = new THREE.Vector3();
const target = new THREE.Vector3();
const followCamera = new THREE.Vector3();
let heading = 0,
  moving = false,
  pointerActive = false,
  zoom = 20,
  spawnClock = 0,
  collisionClock = 0;
const pointer = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
const floor = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const rayHit = new THREE.Vector3();
let species = [0, 0, 0];
let traveled = 0;
let lastMilestone = 0;
function makePikmin(x, z, type) {
  return {
    x,
    z,
    type,
    seed: Math.random() * TAU,
    angle: Math.random() * TAU,
    joined: elapsed,
  };
}
function recruit(p) {
  p.joined = elapsed;
  crowd.push(p);
  species[p.type]++;
  if (elapsed > 0.1) welcome(p);
}
function cluster(x, z, n) {
  for (let i = 0; i < n; i++) {
    const a = rand(0, TAU),
      r = Math.sqrt(Math.random()) * Math.max(1.1, Math.sqrt(n) * 0.24);
    wild.push(
      makePikmin(
        x + Math.cos(a) * r,
        z + Math.sin(a) * r,
        Math.floor(Math.random() * 3),
      ),
    );
  }
}
function reset() {
  for (const batch of batches) {
    if (batch.mesh) {
      scene.remove(batch.mesh);
      batch.mesh.dispose();
    }
    batch.mesh = null;
    batch.capacity = 0;
    batch.count = 0;
  }
  crowd = [];
  wild = [];
  species = [0, 0, 0];
  player.set(0, 0, 0);
  target.copy(player);
  followCamera.copy(player);
  pointerActive = false;
  elapsed = 0;
  for (const s of sparks) s.born = -10;
  traveled = 0;
  lastMilestone = 0;
  zoom = 20;
  heading = 0;
  spawnClock = 0;
  collisionClock = 0;
  for (let i = 0; i < 5; i++)
    recruit(makePikmin(rand(-1, 1), rand(0.6, 2), i % 3));
  for (let i = 0; i < 24; i++) {
    const a = i * 2.4,
      r = 4 + Math.sqrt(i) * 3.7;
    cluster(Math.cos(a) * r, Math.sin(a) * r, Math.floor(rand(3, 9)));
  }
  refreshCount();
  $("hint").style.opacity = "1";
}
function refreshCount() {
  $("count").textContent = crowd.length.toLocaleString();
}

// Endless set dressing recycles around the camera, independent of crowd growth.
const deco = new THREE.Group();
scene.add(deco);
const decorations = [];
function cloverGeometry() {
  const p = [];
  for (let i = 0; i < 3; i++) {
    const a = (i * TAU) / 3;
    p.push([
      sphere,
      "#63834d",
      [Math.sin(a) * 0.21, 0.65, Math.cos(a) * 0.21],
      [0.23, 0.055, 0.28],
      [0.1, a, 0.1],
    ]);
  }
  p.push([
    new THREE.CylinderGeometry(0.024, 0.035, 0.65, 5),
    "#7d9150",
    [0, 0.325, 0],
    [1, 1, 1],
  ]);
  return mergeParts(p);
}
function flowerGeometry() {
  const p = [
    [
      new THREE.CylinderGeometry(0.03, 0.05, 1.8, 5),
      "#6e8945",
      [0, 0.9, 0],
      [1, 1, 1],
    ],
  ];
  for (let i = 0; i < 7; i++) {
    const a = (i * TAU) / 7;
    p.push([
      sphere,
      "#faf2d9",
      [Math.cos(a) * 0.3, 1.8, Math.sin(a) * 0.3],
      [0.27, 0.075, 0.15],
      [0, -a, 0],
    ]);
  }
  p.push([sphere, "#e5b84c", [0, 1.84, 0], [0.17, 0.09, 0.17]]);
  return mergeParts(p);
}
function mushroomGeometry() {
  return mergeParts([
    [sphere, "#e5d8b4", [0, 0.4, 0], [0.15, 0.4, 0.15]],
    [
      new THREE.SphereGeometry(1, 12, 6, 0, TAU, 0, Math.PI / 2),
      "#b86142",
      [0, 0.7, 0],
      [0.53, 0.3, 0.53],
    ],
    [sphere, "#f0dcba", [-0.17, 0.94, 0.12], [0.1, 0.015, 0.075]],
    [sphere, "#f0dcba", [0.22, 0.9, 0.17], [0.075, 0.016, 0.075]],
    [sphere, "#f0dcba", [0.1, 0.98, -0.12], [0.07, 0.015, 0.07]],
  ]);
}
function strawberryGeometry() {
  const p = [[sphere, "#cc4438", [0, 0.65, 0], [0.57, 0.7, 0.54]]];
  for (let i = 0; i < 5; i++) {
    const a = (i * TAU) / 5;
    p.push([
      sphere,
      "#567642",
      [Math.cos(a) * 0.23, 1.29, Math.sin(a) * 0.23],
      [0.29, 0.035, 0.12],
      [0, -a, 0.12],
    ]);
  }
  for (let i = 0; i < 32; i++) {
    const y = -0.75 + (i / 32) * 1.55,
      a = i * 2.4,
      r = Math.sqrt(1 - y * y);
    p.push([
      sphere,
      "#f3cd80",
      [Math.cos(a) * r * 0.573, 0.65 + y * 0.7, Math.sin(a) * r * 0.543],
      [0.025, 0.041, 0.025],
    ]);
  }
  return mergeParts(p);
}
function acornGeometry() {
  return mergeParts([
    [sphere, "#a67846", [0, 0.36, 0], [0.29, 0.43, 0.29]],
    [
      new THREE.SphereGeometry(1, 10, 5, 0, TAU, 0, Math.PI / 2),
      "#765b37",
      [0, 0.51, 0],
      [0.34, 0.22, 0.34],
    ],
    [
      new THREE.CylinderGeometry(0.035, 0.045, 0.19, 5),
      "#6b5134",
      [0, 0.81, 0],
      [1, 1, 1],
      [0, 0, 0.3],
    ],
  ]);
}
const decoMat = new THREE.MeshStandardMaterial({
  vertexColors: true,
  roughness: 1,
});
for (const [geo, n, scale] of [
  [cloverGeometry(), 1700, 1],
  [flowerGeometry(), 100, 1],
  [mushroomGeometry(), 80, 1],
  [strawberryGeometry(), 35, 1.5],
  [acornGeometry(), 55, 1.5],
  [new THREE.DodecahedronGeometry(1, 0), 110, 1],
]) {
  const material = geo.attributes.color ? decoMat : mat("#939782");
  const mesh = new THREE.InstancedMesh(geo, material, n);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  deco.add(mesh);
  const items = [];
  for (let i = 0; i < n; i++)
    items.push({
      x: rand(-85, 85),
      z: rand(-85, 85),
      scale: rand(0.4, 1.4) * scale,
      a: rand(0, TAU),
    });
  decorations.push({ mesh, items, rock: !geo.attributes.color });
}
// Grass ribbons and a few windblown seed motes give the clearing a living surface.
const grassGeo = new THREE.BufferGeometry();
grassGeo.setAttribute(
  "position",
  new THREE.Float32BufferAttribute(
    [
      -0.035, 0, 0, 0.035, 0, 0, 0.06, 0.38, 0, -0.035, 0, 0.02, 0.06, 0.38, 0,
      0.09, 0.65, 0,
    ],
    3,
  ),
);
grassGeo.computeVertexNormals();
const grassMat = mat("#859b57", { side: THREE.DoubleSide });
let grassShader;
grassMat.onBeforeCompile = (s) => {
  grassShader = s;
  s.uniforms.time = { value: 0 };
  s.vertexShader =
    "uniform float time;\n" +
    s.vertexShader.replace(
      "#include <begin_vertex>",
      "#include <begin_vertex>\n transformed.x += sin(time * 1.5 + instanceMatrix[3].x * .4 + instanceMatrix[3].z * .3) * position.y * .17;",
    );
};
const grass = new THREE.InstancedMesh(grassGeo, grassMat, 8500);
grass.frustumCulled = false;
scene.add(grass);
const grassItems = Array.from({ length: 8500 }, () => ({
  x: rand(-80, 80),
  z: rand(-80, 80),
  s: rand(0.4, 1.3),
  a: rand(0, TAU),
}));
let decoX = Infinity,
  decoZ = Infinity;
const wrap = (n, size) => ((((n + size / 2) % size) + size) % size) - size / 2;
function updateMeadow(force = false) {
  if (!force && Math.hypot(player.x - decoX, player.z - decoZ) < 6) return;
  decoX = player.x;
  decoZ = player.z;
  for (const { mesh, items, rock } of decorations) {
    items.forEach((p, i) => {
      const x = player.x + wrap(p.x - player.x, 170),
        z = player.z + wrap(p.z - player.z, 170);
      const clear = Math.hypot(x, z) < 3 ? 0 : 1;
      dummy.position.set(x, rock ? 0.18 * p.scale : 0, z);
      dummy.rotation.set(0, p.a, rock ? 0.15 : 0);
      dummy.scale.set(
        p.scale * clear,
        p.scale * (rock ? 0.6 : 1) * clear,
        p.scale * clear,
      );
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  }
  grassItems.forEach((p, i) => {
    dummy.position.set(
      player.x + wrap(p.x - player.x, 160),
      0,
      player.z + wrap(p.z - player.z, 160),
    );
    dummy.rotation.set(0, p.a, 0);
    dummy.scale.setScalar(p.s);
    dummy.updateMatrix();
    grass.setMatrixAt(i, dummy.matrix);
  });
  grass.instanceMatrix.needsUpdate = true;
}
const moteCount = 90,
  motePositions = new Float32Array(moteCount * 3),
  moteSeeds = Array.from({ length: moteCount }, () => [
    rand(-35, 35),
    rand(1, 7),
    rand(-35, 35),
    rand(0, TAU),
  ]);
const moteGeo = new THREE.BufferGeometry();
moteGeo.setAttribute("position", new THREE.BufferAttribute(motePositions, 3));
const motes = new THREE.Points(
  moteGeo,
  new THREE.PointsMaterial({
    color: "#ffffe3",
    size: 0.065,
    transparent: true,
    opacity: 0.7,
  }),
);
motes.frustumCulled = false;
scene.add(motes);
const sparkCanvas = document.createElement("canvas");
sparkCanvas.width = sparkCanvas.height = 32;
const ctx = sparkCanvas.getContext("2d");
const glow = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
glow.addColorStop(0, "#ffffef");
glow.addColorStop(0.2, "#fff6be");
glow.addColorStop(1, "#ffffdd00");
ctx.fillStyle = glow;
ctx.fillRect(0, 0, 32, 32);
const sparks = Array.from({ length: 300 }, () => ({
  x: 0,
  z: 0,
  born: -10,
  a: 0,
}));
let sparkNext = 0;
const sparkPositions = new Float32Array(900);
sparkPositions.fill(-10000);
const sparkGeo = new THREE.BufferGeometry();
sparkGeo.setAttribute("position", new THREE.BufferAttribute(sparkPositions, 3));
const sparkMesh = new THREE.Points(
  sparkGeo,
  new THREE.PointsMaterial({
    map: new THREE.CanvasTexture(sparkCanvas),
    color: "#ffffd6",
    size: 0.19,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  }),
);
sparkMesh.frustumCulled = false;
scene.add(sparkMesh);
function welcome(p) {
  for (let i = 0; i < 3; i++) {
    const s = sparks[sparkNext++ % sparks.length];
    s.x = p.x;
    s.z = p.z;
    s.born = elapsed;
    s.a = rand(0, TAU);
  }
}
function updateWelcome() {
  sparks.forEach((s, i) => {
    const age = elapsed - s.born;
    if (age < 0 || age > 1) {
      sparkPositions[i * 3 + 1] = -10000;
      return;
    }
    sparkPositions[i * 3] = s.x + Math.cos(s.a) * age * 0.6;
    sparkPositions[i * 3 + 1] = 0.6 + age * 1.4;
    sparkPositions[i * 3 + 2] = s.z + Math.sin(s.a) * age * 0.6;
  });
  sparkGeo.attributes.position.needsUpdate = true;
}
const ring = new THREE.Mesh(
  new THREE.RingGeometry(0.28, 0.32, 40),
  new THREE.MeshBasicMaterial({
    color: "#f7f4d8",
    transparent: true,
    opacity: 0.7,
    depthWrite: false,
  }),
);
ring.rotation.x = -Math.PI / 2;
ring.position.y = 0.025;
scene.add(ring);
ring.visible = false;
const beacon = new THREE.Sprite(
  new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(sparkCanvas),
    color: "#f87648",
    transparent: true,
    depthTest: false,
    opacity: 0.8,
  }),
);
scene.add(beacon);

// Spatial hashing checks actual followers, not a radius approximation of the crowd.
const cellSize = 1.2;
const grid = new Map();
const gridKey = (x, z) => Math.imul(x, 73856093) ^ Math.imul(z, 19349663);
function collect() {
  grid.clear();
  for (let i = 0; i < wild.length; i++) {
    const p = wild[i],
      key = gridKey(Math.floor(p.x / cellSize), Math.floor(p.z / cellSize));
    let bucket = grid.get(key);
    if (!bucket) grid.set(key, (bucket = []));
    bucket.push(i);
  }
  const found = new Set();
  function touch(x, z) {
    const cx = Math.floor(x / cellSize),
      cz = Math.floor(z / cellSize);
    for (let a = -1; a <= 1; a++)
      for (let b = -1; b <= 1; b++) {
        const bucket = grid.get(gridKey(cx + a, cz + b));
        if (bucket)
          for (const i of bucket) {
            const p = wild[i];
            if ((p.x - x) ** 2 + (p.z - z) ** 2 < 0.78 ** 2) found.add(i);
          }
      }
  }
  touch(player.x, player.z);
  for (const p of crowd) touch(p.x, p.z);
  if (found.size) {
    for (const i of found) recruit(wild[i]);
    wild = wild.filter((_, i) => !found.has(i));
    refreshCount();
    chirp();
    const milestone = Math.max(
      [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
        .filter((n) => n <= crowd.length)
        .pop() || 0,
      Math.floor(crowd.length / 100000) * 100000,
    );
    if (milestone > lastMilestone) {
      lastMilestone = milestone;
      $("count").animate(
        [{ transform: "scale(1.18)" }, { transform: "scale(1)" }],
        { duration: 450, easing: "ease-out" },
      );
    }
  }
}
function seedWorld(dt, radius) {
  spawnClock += dt;
  if (spawnClock < 0.55) return;
  spawnClock = 0;
  const reach = Math.max(23, radius * 1.9 + 15);
  wild = wild.filter(
    (p) => Math.hypot(p.x - player.x, p.z - player.z) < reach * 1.65,
  );
  const desired = Math.max(180, Math.ceil(crowd.length * 0.22));
  if (wild.length < desired) {
    for (let i = 0; i < 5; i++) {
      const a = rand(0, TAU),
        r = rand(radius + 5, reach);
      cluster(
        player.x + Math.cos(a) * r,
        player.z + Math.sin(a) * r,
        Math.max(4, Math.ceil(Math.sqrt(crowd.length) * rand(0.6, 1.5))),
      );
    }
  }
}
function renderPikmin() {
  const totals = [...species];
  for (const p of wild) totals[p.type]++;
  for (const b of batches) {
    ensureCapacity(b, totals[b.type]);
    b.count = 0;
    b.mesh.geometry = crowd.length > 4000 ? simple[b.type] : detailed[b.type];
  }
  function put(p, joined) {
    const b = batches[p.type];
    const bounce = joined
      ? Math.abs(Math.sin(elapsed * (moving ? 14 : 3) + p.seed)) *
        (moving ? 0.11 : 0.022)
      : Math.sin(elapsed * 2 + p.seed) * 0.015;
    dummy.position.set(p.x, bounce, p.z);
    dummy.rotation.set(0, p.angle, Math.sin(elapsed * 4 + p.seed) * 0.045);
    const pop = joined
      ? 1 + Math.sin(Math.min(1, (elapsed - p.joined) * 3) * Math.PI) * 0.3
      : 1;
    dummy.scale.setScalar(pop);
    dummy.updateMatrix();
    b.mesh.setMatrixAt(b.count++, dummy.matrix);
  }
  for (const p of crowd) put(p, true);
  for (const p of wild) put(p, false);
  for (const b of batches) {
    b.mesh.count = b.count;
    b.mesh.instanceMatrix.needsUpdate = true;
  }
}
let audioContext,
  soundEnabled = false;
function chirp() {
  if (!soundEnabled) return;
  try {
    audioContext ??= new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === "suspended") audioContext.resume();
    const t = audioContext.currentTime;
    const oscillator = audioContext.createOscillator(),
      gain = audioContext.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(660 + Math.random() * 440, t);
    oscillator.frequency.exponentialRampToValueAtTime(1400, t + 0.08);
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.035, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.13);
    oscillator.connect(gain);
    gain.connect(audioContext.destination);
    oscillator.start(t);
    oscillator.stop(t + 0.14);
  } catch {
    soundEnabled = false;
  }
}
function toggleSound() {
  soundEnabled = !soundEnabled;
  $("sound").setAttribute(
    "aria-label",
    soundEnabled ? "Turn sound off" : "Turn sound on",
  );
  $("sound-wave").setAttribute(
    "d",
    soundEnabled ? "M16 8q5 4 0 8M18 5q8 7 0 14" : "m16 9 5 6m0-6-5 6",
  );
  if (soundEnabled) chirp();
}
let renderDirty = true;
function setPaused(value) {
  paused = value;
  renderDirty = true;
  $("pause").setAttribute("aria-label", paused ? "Resume" : "Pause");
  $("pause").title = paused ? "Resume (Space)" : "Pause (Space)";
  $("pause").innerHTML = paused
    ? '<svg viewBox="0 0 24 24"><path d="m8 5 11 7-11 7z"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="M8 5v14M16 5v14"/></svg>';
  if (paused) {
    pointerActive = false;
    ring.visible = false;
  }
}
const canvas = renderer.domElement;
if (matchMedia("(pointer: coarse)").matches)
  $("hint").lastElementChild.textContent = "Drag to lead.";
function movePointer(event) {
  if (paused || $("about").open) return;
  pointer.set(
    (event.clientX / innerWidth) * 2 - 1,
    (-event.clientY / innerHeight) * 2 + 1,
  );
  pointerActive = true;
}
canvas.addEventListener("pointermove", (event) => {
  if (event.pointerType === "mouse" || event.buttons) movePointer(event);
});
canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  movePointer(event);
});
canvas.addEventListener("pointerup", (event) => {
  if (event.pointerType !== "mouse") {
    pointerActive = false;
    ring.visible = false;
  }
});
canvas.addEventListener("pointercancel", () => {
  pointerActive = false;
  ring.visible = false;
});
canvas.addEventListener("pointerleave", () => {
  pointerActive = false;
  ring.visible = false;
});
$("sound").onclick = toggleSound;
$("pause").onclick = () => setPaused(!paused);
$("info").onclick = () => {
  wasPaused = paused;
  setPaused(true);
  $("about").showModal();
};
$("close").onclick = () => $("about").close();
$("about").addEventListener("close", () => setPaused(wasPaused));
$("restart").onclick = () => {
  reset();
  updateMeadow(true);
  wasPaused = false;
  $("about").close();
};
window.addEventListener("keydown", (event) => {
  if (event.target.closest("button,dialog,a")) return;
  if (event.code === "Space") {
    event.preventDefault();
    setPaused(!paused);
  }
  if (event.code === "KeyM") toggleSound();
});
window.addEventListener("blur", () => {
  pointerActive = false;
  ring.visible = false;
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    pointerActive = false;
    ring.visible = false;
  }
});
canvas.addEventListener("webglcontextlost", (event) => {
  event.preventDefault();
  running = false;
  $("loading").style.display = "flex";
  $("loading").style.opacity = "1";
  $("loading").innerHTML =
    '<p>The meadow needs a moment.</p><p style="font:13px sans-serif">Reload to begin a fresh expedition.</p>';
});
window.addEventListener("resize", () => {
  renderDirty = true;
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
});
function lerpAngle(a, b, t) {
  return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;
}
let previous = performance.now();
function animate(now) {
  if (!running) return;
  requestAnimationFrame(animate);
  const dt = Math.min((now - previous) / 1000, 0.05);
  previous = now;
  if ((paused || document.hidden) && !renderDirty) return;
  renderDirty = false;
  const radius = Math.sqrt(crowd.length) * 0.32 + 1.0;
  if (!paused && !document.hidden) {
    elapsed += dt;
    if (pointerActive) {
      raycaster.setFromCamera(pointer, camera);
      if (raycaster.ray.intersectPlane(floor, rayHit)) {
        target.copy(rayHit);
        ring.position.set(target.x, 0.025, target.z);
        ring.visible = true;
      }
    }
    const dx = target.x - player.x,
      dz = target.z - player.z,
      distance = Math.hypot(dx, dz);
    moving = pointerActive && distance > 0.35;
    if (moving) {
      const speed = 5.5 + Math.sqrt(crowd.length) * 0.06,
        step = Math.min(distance - 0.25, speed * dt);
      player.x += (dx / distance) * step;
      player.z += (dz / distance) * step;
      traveled += step;
      if (traveled > 3) $("hint").style.opacity = "0";
      heading = lerpAngle(heading, Math.atan2(dx, dz), Math.min(1, dt * 10));
    }
    captain.position.copy(player);
    captain.rotation.y = heading;
    body.position.y = moving
      ? Math.abs(Math.sin(elapsed * 11)) * 0.065
      : Math.sin(elapsed * 2) * 0.014;
    limbs.forEach(
      (limb, i) =>
        (limb.rotation.x = moving
          ? Math.sin(elapsed * 11 + (i === 0 || i === 3 ? 0 : Math.PI)) * 0.45
          : 0),
    );
    antenna.material.emissiveIntensity = 0.5 + Math.sin(elapsed * 3) * 0.15;
    const relax = 1 - Math.exp(-dt * 4);
    // Golden-angle packing keeps a stable, dense following in linear time.
    for (let i = 0; i < crowd.length; i++) {
      const p = crowd[i],
        a = i * 2.399963229728653,
        r = 0.34 * Math.sqrt(i + 1);
      const tx =
          player.x +
          Math.cos(a) * r +
          Math.sin(p.seed) * 0.13 -
          Math.sin(heading) * Math.min(radius * 0.42, 4),
        tz =
          player.z +
          Math.sin(a) * r +
          Math.cos(p.seed) * 0.13 -
          Math.cos(heading) * Math.min(radius * 0.42, 4);
      const vx = tx - p.x,
        vz = tz - p.z;
      p.x += vx * relax;
      p.z += vz * relax;
      if (vx * vx + vz * vz > 0.015)
        p.angle = lerpAngle(p.angle, Math.atan2(vx, vz), Math.min(1, dt * 9));
    }
    collisionClock += dt;
    if (collisionClock > 0.12) {
      collect();
      collisionClock = 0;
    }
    seedWorld(dt, radius);
    updateMeadow();
    if (grassShader) grassShader.uniforms.time.value = elapsed;
    if (pikShader) {
      pikShader.uniforms.pikTime.value = elapsed;
      pikShader.uniforms.walking.value = moving ? 1 : 0.1;
    }
    updateWelcome();
    for (let i = 0; i < moteCount; i++) {
      const s = moteSeeds[i];
      motePositions[i * 3] = player.x + wrap(s[0] + elapsed * 0.13, 70);
      motePositions[i * 3 + 1] = s[1] + Math.sin(elapsed * 0.4 + s[3]) * 0.4;
      motePositions[i * 3 + 2] =
        player.z + wrap(s[2] + Math.sin(elapsed * 0.2 + s[3]), 70);
    }
    moteGeo.attributes.position.needsUpdate = true;
  }
  const desiredZoom = (16 + radius * 2.4) * Math.max(1, 0.85 / camera.aspect);
  zoom = THREE.MathUtils.lerp(zoom, desiredZoom, 1 - Math.exp(-dt * 1.4));
  beacon.position.set(player.x + 0.4, 2.42, player.z);
  beacon.scale.setScalar(Math.max(0.24, zoom * 0.014));
  followCamera.lerp(player, 1 - Math.exp(-dt * 3));
  camera.position.set(followCamera.x, zoom, followCamera.z + zoom * 0.98);
  camera.lookAt(followCamera.x, 0, followCamera.z);
  camera.far = Math.max(1500, zoom * 5);
  camera.updateProjectionMatrix();
  scene.fog.near = zoom * 2;
  scene.fog.far = zoom * 4;
  ground.position.set(player.x, 0, player.z);
  ground.scale.setScalar(Math.max(1, zoom / 500));
  sun.position.set(player.x - 18, 32, player.z + 15);
  sun.target.position.copy(player);
  renderPikmin();
  renderer.render(scene, camera);
}
reset();
updateMeadow(true);
camera.position.set(0, zoom, zoom * 0.98);
camera.lookAt(0, 0, 0);
renderPikmin();
renderer.render(scene, camera);
$("loading").style.opacity = "0";
setTimeout(() => ($("loading").style.display = "none"), 800);
requestAnimationFrame(animate);
// Read-only diagnostics for reproducible local performance checks; no player-facing overlay.
window.__meadow = {
  get stats() {
    return {
      crowd: crowd.length,
      wild: wild.length,
      capacity: batches.reduce((n, b) => n + b.capacity, 0),
      drawCalls: renderer.info.render.calls,
      triangles: renderer.info.render.triangles,
      paused,
      player: { x: player.x, z: player.z },
      zoom,
    };
  },
};
