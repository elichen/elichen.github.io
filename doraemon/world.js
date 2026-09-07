import * as THREE from "three";
import { ball, box, cylinder, tube, material } from "./model.js";

let seed = 1909;
function random() {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
}
const range = (a, b) => a + (b - a) * random();
function mergeStatic(group) {
  // Bake the little town into one draw per material, keeping the diorama light on mobile.
  group.updateMatrixWorld(true);
  const batches = new Map();
  group.traverse((obj) => {
    if (!obj.isMesh || Array.isArray(obj.material)) return;
    const key = obj.material;
    if (!batches.has(key)) batches.set(key, []);
    const geo = obj.geometry.clone();
    geo.applyMatrix4(obj.matrixWorld);
    batches.get(key).push(geo);
  });
  const merged = new THREE.Group();
  for (const [mat, geos] of batches) {
    let count = 0,
      indexCount = 0;
    for (const g of geos) {
      count += g.attributes.position.count;
      indexCount += g.index ? g.index.count : g.attributes.position.count;
    }
    const p = new Float32Array(count * 3),
      n = new Float32Array(count * 3),
      uv = new Float32Array(count * 2),
      indices = new Uint32Array(indexCount);
    let vertex = 0,
      index = 0;
    for (const g of geos) {
      p.set(g.attributes.position.array, vertex * 3);
      n.set(g.attributes.normal.array, vertex * 3);
      if (g.attributes.uv) uv.set(g.attributes.uv.array, vertex * 2);
      for (let i = 0; i < (g.index?.count ?? g.attributes.position.count); i++)
        indices[index++] = (g.index ? g.index.array[i] : i) + vertex;
      vertex += g.attributes.position.count;
      g.dispose();
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("normal", new THREE.BufferAttribute(n, 3));
    g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    g.setIndex(new THREE.BufferAttribute(indices, 1));
    const m = new THREE.Mesh(g, mat);
    m.castShadow = true;
    m.receiveShadow = true;
    merged.add(m);
  }
  return merged;
}
export function makeWorld(scene) {
  seed = 1909;
  const skyUniforms = {
    top: { value: new THREE.Color("#258dca") },
    bottom: { value: new THREE.Color("#a9d8ef") },
    sunColor: { value: new THREE.Color("#fff5cf") },
    sunDir: { value: new THREE.Vector3(-0.74, 0.11, -1).normalize() },
    night: { value: 0 },
    time: { value: 0 },
  };
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(450, 32, 20),
    new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: skyUniforms,
      vertexShader: `varying vec3 vDir;void main(){vDir=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
      fragmentShader: `
        varying vec3 vDir;
        uniform vec3 top, bottom, sunColor, sunDir;
        uniform float night, time;
        float hash(vec3 p) { return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453); }
        void main() {
          vec3 d = normalize(vDir);
          float height = pow(max(d.y + .055, 0.), .54);
          vec3 col = mix(bottom, top, clamp(height, 0., 1.));
          float sun = max(dot(d, sunDir), 0.);
          col += sunColor * (pow(sun, 12.) * .09 + pow(sun, 90.) * .085) * mix(1., .18, night);
          float disc = smoothstep(.99930, .99946, sun);
          vec3 moonShadow = normalize(sunDir + vec3(.018, .009, .001));
          float crescent = 1. - smoothstep(.99932, .99948, dot(d, moonShadow));
          col = mix(col, sunColor * 1.15, disc * mix(1., crescent, night));
          // A few long brushstrokes of cirrus soften the otherwise perfect dome.
          vec2 q = d.xz / max(d.y + .22, .08);
          float wisps = sin(q.x * 2.2 + q.y * 1.8 + sin(q.y * 1.7) * 1.8);
          wisps *= sin(q.x * .85 - q.y * .4);
          float veil = pow(max(wisps, 0.), 7.) * smoothstep(.08, .32, d.y);
          col = mix(col, bottom, veil * .17 * (1. - night));
          vec3 cell = floor(d * 380.);
          float star = step(.9963, hash(cell)) * pow(max(0., 1. - length(fract(d * 380.) - .5) * 2.), 3.);
          col += star * night * 3.2 * (.72 + .28 * sin(time * .7 + hash(cell) * 30.));
          // The faint, angled band makes the night sky feel deep without a texture.
          float galaxy = pow(max(0., 1. - abs(d.x * .52 + d.y * .65 + d.z * .2 - .18) * 6.), 3.);
          col += vec3(.038, .042, .072) * galaxy * night * smoothstep(0., .3, d.y);
          gl_FragColor = vec4(col, 1.);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }),
  );
  sky.renderOrder = -10;
  scene.add(sky);
  const seaUniforms = {
    time: { value: 0 },
    water: { value: new THREE.Color("#42acbc") },
    horizon: { value: new THREE.Color("#a9d8ef") },
    shine: { value: new THREE.Color("#daf7ed") },
    night: { value: 0 },
    sunDir: skyUniforms.sunDir,
    skyTop: skyUniforms.top,
    sunColor: skyUniforms.sunColor,
    shores: { value: [
      new THREE.Vector4(-4, -12, 8.8, .25),
      new THREE.Vector4(-40, 8, 7.15, .38),
      new THREE.Vector4(-19, 29, 5.9, .3),
      new THREE.Vector4(33, -32, 4.6, .24),
    ] },
  };
  const sea = new THREE.Mesh(
    new THREE.PlaneGeometry(1200, 1200, 1, 1),
    new THREE.ShaderMaterial({
      uniforms: seaUniforms,
      vertexShader: `varying vec3 vWorld;void main(){vec4 w=modelMatrix*vec4(position,1.);vWorld=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
      fragmentShader: `
        varying vec3 vWorld;
        uniform float time, night;
        uniform vec3 water, horizon, shine, sunDir, skyTop, sunColor;
        uniform vec4 shores[4];
        void main() {
          vec2 p = vWorld.xz;
          vec3 eye = normalize(cameraPosition - vWorld);
          float swell = sin(p.x * .22 + p.y * .17 + time * .33);
          float smallWave = sin(p.y * 2.6 + sin(p.x * .38 + time * .2) * .7 + time * .8);
          vec3 normal = normalize(vec3(
            cos(p.x * .22 + p.y * .17 + time * .33) * .065 + cos(p.x * .68 + time * .4) * .025,
            1., cos(p.y * .54 + time * .55) * .055 + smallWave * .02));
          float fresnel = pow(1. - max(dot(eye, normal), 0.), 3.);
          vec3 col = mix(water * (.97 + swell * .025), horizon, fresnel * .38);
          float ripple = smoothstep(.89, 1., smallWave) * pow(max(0., sin(p.x * .62 + sin(p.y * .24))), 8.);
          col = mix(col, shine, ripple * .12);
          vec3 halfVector = normalize(eye + sunDir);
          float specular = pow(max(dot(normal, halfVector), 0.), 150.);
          col += shine * specular * .14;
          // Turquoise lagoons and delicate broken surf share one ocean draw call.
          for (int i = 0; i < 4; i++) {
            vec2 delta = p - shores[i].xy;
            float angle = atan(delta.y, delta.x);
            float shore = length(delta) - shores[i].z - sin(angle * 7.) * shores[i].w;
            float shallow = (1. - smoothstep(0., 4.2, shore)) * smoothstep(-1.3, .15, shore);
            col = mix(col, water * vec3(.86, 1.19, 1.13), shallow * .57);
            float wash = sin(shore * 4.8 - time * .62 + sin(angle * 5.) * .22);
            float foam = smoothstep(.89, .98, wash) * (1. - smoothstep(.4, 2.4, shore));
            foam *= smoothstep(-.2, .3, shore) * (.57 + .43 * sin(angle * 13. + time * .16));
            col = mix(col, shine, max(0., foam) * .48);
          }
          float dist = length(cameraPosition.xz - p);
          vec3 horizonColor = mix(horizon, skyTop, pow(max(.055 - eye.y, 0.), .54));
          float sun = max(dot(-eye, sunDir), 0.);
          horizonColor += sunColor * (pow(sun, 12.) * .09 + pow(sun, 90.) * .085) * mix(1., .18, night);
          col = mix(col, horizonColor, smoothstep(85., 310., dist));
          gl_FragColor = vec4(col, 1.);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    }),
  );
  sea.rotation.x = -Math.PI / 2;
  sea.position.y = -5.3;
  scene.add(sea);
  const diorama = new THREE.Group();
  diorama.scale.setScalar(0.62);
  diorama.position.set(-4, -2.014, -12);
  scene.add(diorama);
  const staticRoot = new THREE.Group();
  const sand = material("#efd7a4", { roughness: 1 }),
    grass = material("#91ba65", { roughness: 1 }),
    rock = material("#b1b4a0", { roughness: 1 }),
    trunk = material("#957b55", { roughness: 1 }),
    leaf = material("#62945b", { roughness: 0.85 }),
    leafLight = material("#91b76d", { roughness: 1 }),
    darkLeaf = material("#397b6b", { roughness: 1 });
  const wall = material("#fff1d3", { roughness: 0.9 }),
    peach = material("#eed3b5", { roughness: 1 }),
    roof = material("#ce786b", { roughness: 0.8 }),
    blueRoof = material("#698f9c", { roughness: 0.8 }),
    windowMat = material("#759daf", { roughness: 0.32 }),
    wood = material("#b58b5d", { roughness: 1 }),
    cream = material("#fff6df", { roughness: 0.7 }),
    red = material("#da8473", { roughness: 0.8 }),
    pathMat = material("#d9cb9f", { roughness: 1 });
  const groundHeight = (x, z) =>
    -4.4 +
    1.65 * Math.max(0, 1 - (Math.hypot(x, z) / 13) ** 2) +
    0.36 * Math.sin(x * 0.42) * Math.cos(z * 0.35);
  function land(radius, mat, offset = 0) {
    const vertices = [],
      inds = [],
      rings = 22,
      seg = 100;
    for (let j = 0; j <= rings; j++)
      for (let i = 0; i <= seg; i++) {
        const a = (i / seg) * Math.PI * 2;
        const edge = 1 + 0.035 * Math.sin(a * 7) + 0.018 * Math.cos(a * 11);
        const r = (j / rings) * radius * edge,
          x = Math.cos(a) * r,
          z = Math.sin(a) * r;
        vertices.push(x, groundHeight(x, z) + offset, z);
      }
    for (let j = 0; j < rings; j++)
      for (let i = 0; i < seg; i++) {
        const a = j * (seg + 1) + i;
        inds.push(a, a + 1, a + seg + 1, a + 1, a + seg + 2, a + seg + 1);
      }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geo.setIndex(inds);
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    staticRoot.add(mesh);
  }
  ball(staticRoot, sand, 0, -5.15, 0, 14.2, 1.05, 13.7, true);
  land(13.6, sand);
  land(12.4, grass, 0.06);
  // A winding path from the seaside village to the lighthouse.
  const pathPoints = [];
  for (let i = 0; i <= 35; i++) {
    const z = -9 + (i / 35) * 18,
      x = Math.sin(z * 0.28) * 2.1;
    pathPoints.push([x, groundHeight(x, z) + 0.095, z]);
  }
  tube(staticRoot, pathPoints, 0.3, pathMat);
  function tree(x, z, size = 1, pine = false) {
    const y = groundHeight(x, z),
      g = new THREE.Group();
    g.position.set(x, y, z);
    staticRoot.add(g);
    cylinder(
      g,
      trunk,
      0.075 * size,
      0.12 * size,
      1 * size,
      0,
      0.5 * size,
      0,
      8,
    );
    if (pine) {
      for (let i = 0; i < 3; i++) {
        const m = new THREE.Mesh(
          new THREE.ConeGeometry((0.7 - i * 0.13) * size, 1.05 * size, 9),
          i === 2 ? leafLight : darkLeaf,
        );
        m.position.y = (0.9 + i * 0.42) * size;
        m.castShadow = true;
        g.add(m);
      }
    } else {
      ball(
        g,
        leaf,
        0,
        1.25 * size,
        0,
        0.7 * size,
        0.82 * size,
        0.64 * size,
        true,
      );
      ball(
        g,
        leafLight,
        0.28 * size,
        1.43 * size,
        0.08 * size,
        0.47 * size,
        0.63 * size,
        0.5 * size,
        true,
      );
    }
  }
  function house(x, z, size = 1, blue = false, turn = 0) {
    const y = groundHeight(x, z) + 0.1,
      g = new THREE.Group();
    g.position.set(x, y, z);
    g.rotation.y = turn;
    g.scale.setScalar(size);
    staticRoot.add(g);
    box(g, random() > 0.5 ? wall : peach, 0, 0.61, 0, 1.5, 1.2, 1.25);
    const shape = new THREE.Shape();
    shape.moveTo(-0.91, 0);
    shape.lineTo(0, 0.66);
    shape.lineTo(0.91, 0);
    shape.closePath();
    const geo = new THREE.ExtrudeGeometry(shape, {
      depth: 1.58,
      bevelEnabled: false,
    });
    const r = new THREE.Mesh(geo, blue ? blueRoof : roof);
    r.position.set(0, 1.2, -0.79);
    g.add(r);
    box(g, wood, 0.3, 0.33, 0.637, 0.29, 0.66, 0.035);
    box(g, cream, -0.37, 0.73, 0.655, 0.5, 0.45, 0.05);
    box(g, windowMat, -0.37, 0.73, 0.687, 0.39, 0.34, 0.026);
    box(g, cream, -0.37, 0.73, 0.71, 0.035, 0.34, 0.025);
    box(g, cream, -0.37, 0.73, 0.71, 0.39, 0.035, 0.025);
    box(g, cream, 0.765, 0.75, 0, 0.035, 0.46, 0.5);
    box(g, windowMat, 0.79, 0.75, 0, 0.018, 0.36, 0.39);
    box(g, cream, 0.8, 0.75, 0, 0.018, 0.035, 0.4);
    box(g, wall, 0.48, 1.75, -0.28, 0.22, 0.6, 0.25);
    box(g, pathMat, 0.28, 0.025, 1.02, 0.52, 0.05, 0.7);
    for (let i = 0; i < 5; i++)
      ball(
        g,
        i % 2 ? roof : cream,
        -0.58 + i * 0.16,
        0.08,
        0.77,
        0.068,
        0.11,
        0.067,
        true,
      );
  }
  house(-4, 1, 1.25, false, 0.2);
  house(-6, -1.8, 0.9, true, -0.2);
  house(-3.7, -3.3, 0.85, false, 0.13);
  house(3.3, -3.7, 1.05, true, -0.35);
  house(5.8, -1, 0.85, false, -0.4);
  house(3.8, 2.6, 0.72, false, -0.2);
  for (let i = 0; i < 47; i++) {
    const a = range(0, Math.PI * 2),
      r = range(7.3, 11.2);
    tree(Math.cos(a) * r, Math.sin(a) * r, range(0.6, 1.25), i % 4 === 0);
  }
  tree(-2.6, 3.5, 1.25);
  tree(-5, 4, 1);
  tree(5, -5, 1.2);
  tree(6.3, 3.3, 0.8);
  // White-and-coral lighthouse, with a glass lantern and balcony rail.
  const lighthouse = new THREE.Group();
  const lx = 8.5,
    lz = 5;
  lighthouse.position.set(lx, groundHeight(lx, lz), lz);
  staticRoot.add(lighthouse);
  cylinder(lighthouse, cream, 0.48, 0.71, 3.4, 0, 1.7, 0);
  cylinder(lighthouse, red, 0.58, 0.62, 0.52, 0, 1.23, 0);
  cylinder(lighthouse, red, 0.505, 0.55, 0.48, 0, 2.34, 0);
  cylinder(lighthouse, cream, 0.78, 0.78, 0.13, 0, 3.4, 0);
  cylinder(lighthouse, windowMat, 0.42, 0.42, 0.73, 0, 3.84, 0);
  cylinder(lighthouse, red, 0.05, 0.66, 0.45, 0, 4.39, 0);
  ball(lighthouse, goldMat(), 0, 4.66, 0, 0.075);
  for (let i = 0; i < 12; i++) {
    const a = (i / 12) * Math.PI * 2;
    cylinder(
      lighthouse,
      cream,
      0.025,
      0.025,
      0.43,
      Math.cos(a) * 0.7,
      3.66,
      Math.sin(a) * 0.7,
      6,
    );
  }
  for (const h of [3.65, 3.87]) {
    const r = new THREE.Mesh(new THREE.TorusGeometry(0.7, 0.025, 6, 32), cream);
    r.rotation.x = Math.PI / 2;
    r.position.y = h;
    lighthouse.add(r);
  }
  box(lighthouse, wood, 0, 0.42, 0.65, 0.32, 0.81, 0.05);
  // Wooden jetty and its tiny moored sailboat.
  for (let i = 0; i < 17; i++)
    box(staticRoot, wood, -3, -4.53, 11 + i * 0.27, 1.23, 0.09, 0.23);
  for (const x of [-3.5, -2.5])
    for (const z of [11.5, 13.2, 15.2])
      cylinder(staticRoot, trunk, 0.07, 0.085, 1, x, -4.67, z, 8);
  const boat = new THREE.Group();
  boat.position.set(-0.7, -5.05, 15);
  boat.rotation.y = -0.4;
  diorama.add(boat);
  ball(boat, cream, 0, 0, 0, 0.49, 0.24, 1.05, true);
  ball(boat, wood, 0, 0.1, 0, 0.35, 0.06, 0.82, true);
  cylinder(boat, wood, 0.024, 0.032, 2.3, 0, 1.16, 0, 8);
  const sailShape = new THREE.Shape();
  sailShape.moveTo(0.04, 0.4);
  sailShape.lineTo(0.04, 2.21);
  sailShape.lineTo(0.89, 0.47);
  sailShape.closePath();
  boat.add(
    new THREE.Mesh(
      new THREE.ShapeGeometry(sailShape),
      new THREE.MeshStandardMaterial({
        color: "#fff4d9",
        side: THREE.DoubleSide,
        roughness: 1,
      }),
    ),
  );
  // A pink Anywhere Door hidden on a hill.
  const door = new THREE.Group();
  door.position.set(-6, groundHeight(-6, 6), 6);
  door.rotation.y = 0.5;
  staticRoot.add(door);
  box(door, red, 0, 0.89, 0, 0.84, 1.78, 0.14);
  box(door, peach, 0, 0.88, 0.078, 0.66, 1.57, 0.035);
  box(door, red, 0, 0.9, 0.103, 0.53, 1.36, 0.018);
  ball(door, goldMat(), 0.18, 0.86, 0.14, 0.047);
  // Grass tufts, stepping stones, and fence posts give the miniature a handmade scale.
  for (let i = 0; i < 70; i++) {
    const a = range(0, Math.PI * 2),
      r = range(4, 12),
      x = Math.cos(a) * r,
      z = Math.sin(a) * r;
    if (i % 3 === 0)
      ball(
        staticRoot,
        rock,
        x,
        groundHeight(x, z) + 0.03,
        z,
        range(0.12, 0.3),
        range(0.08, 0.17),
        range(0.14, 0.28),
        true,
      );
    else {
      const s = range(0.1, 0.22);
      const m = new THREE.Mesh(new THREE.ConeGeometry(s, s * 2, 4), leafLight);
      m.position.set(x, groundHeight(x, z) + s, z);
      staticRoot.add(m);
    }
  }
  for (let i = 0; i < 9; i++) {
    const x = -6.2 + i * 0.4,
      z = 2.4;
    cylinder(
      staticRoot,
      cream,
      0.035,
      0.035,
      0.45,
      x,
      groundHeight(x, z) + 0.23,
      z,
      6,
    );
  }
  tube(
    staticRoot,
    [
      [-6.2, groundHeight(-6.2, 2.4) + 0.34, 2.4],
      [-4.6, groundHeight(-4.6, 2.4) + 0.34, 2.4],
      [-3, groundHeight(-3, 2.4) + 0.34, 2.4],
    ],
    0.025,
    cream,
  );
  const island = mergeStatic(staticRoot);
  diorama.add(island);
  const beacon = new THREE.Group();
  beacon.position.set(lx, groundHeight(lx, lz) + 3.84, lz);
  diorama.add(beacon);
  const beaconGeo = new THREE.ConeGeometry(4, 23, 32, 1, true);
  beaconGeo.translate(0, -11.5, 0);
  beaconGeo.rotateX(-Math.PI / 2);
  const beamMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    uniforms: { strength: { value: 0 } },
    vertexShader: `varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
    fragmentShader: `varying vec2 vUv;uniform float strength;void main(){float fade=pow(vUv.y,1.4);gl_FragColor=vec4(1.,.91,.64,fade*strength);}`,
  });
  beacon.add(new THREE.Mesh(beaconGeo, beamMat));
  const lamp = new THREE.PointLight("#ffd08a", 0, 8, 2);
  lamp.position.copy(beacon.position);
  diorama.add(lamp);

  // Two neighboring islands turn the flight course into a place to explore.
  // Their buildings are baked by material; only the windmill sails move.
  const remoteRoot = new THREE.Group();
  const blossom = material("#eab6c2", { roughness: 1 });
  const blossomLight = material("#f9d8d4", { roughness: 1 });
  const lavender = material("#a89ac7", { roughness: 1 });
  const gardenGlow = material("#ffecb1", {
    roughness: .8, emissive: "#ffcb75", emissiveIntensity: 0,
  });
  function littleIsland(x, z, radius, rise) {
    const g = new THREE.Group();
    g.position.set(x, -5.3, z);
    remoteRoot.add(g);
    const height = (px, pz) => .18 + rise * Math.pow(Math.max(0, 1 - Math.pow(Math.hypot(px, pz) / radius, 2)), 1.2)
      + Math.sin(px * .7) * Math.cos(pz * .52) * .16;
    ball(g, rock, 0, -.4, 0, radius * .96, .85, radius * .95, true);
    for (const [size, mat, offset] of [[radius, sand, 0], [radius * .89, grass, .045]]) {
      const vertices = [], indices = [], segments = 72, rows = 18;
      for (let j = 0; j <= rows; j++) for (let i = 0; i <= segments; i++) {
        const angle = i / segments * Math.PI * 2;
        const r = j / rows * size * (1 + .026 * Math.sin(angle * 7) + .012 * Math.cos(angle * 11));
        const px = Math.cos(angle) * r, pz = Math.sin(angle) * r;
        vertices.push(px, height(px, pz) + offset, pz);
      }
      for (let j = 0; j < rows; j++) for (let i = 0; i < segments; i++) {
        const a = j * (segments + 1) + i;
        indices.push(a, a + 1, a + segments + 1, a + 1, a + segments + 2, a + segments + 1);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();
      g.add(new THREE.Mesh(geometry, mat));
    }
    return { root: g, height };
  }
  const windmillIsle = littleIsland(-40, 8, 7.15, 4.1);
  const mill = new THREE.Group();
  mill.position.y = windmillIsle.height(0, 0);
  windmillIsle.root.add(mill);
  cylinder(mill, wall, .62, .94, 3.4, 0, 1.7, 0, 24);
  cylinder(mill, blueRoof, 0, 1.04, 1.05, 0, 3.93, 0, 24);
  box(mill, wood, 0, .49, .86, .4, .95, .06);
  for (const side of [-1, 1]) {
    box(mill, cream, side * .52, 1.39, .62, .27, .43, .08);
    box(mill, windowMat, side * .52, 1.39, .67, .2, .33, .045);
  }
  const millRotor = new THREE.Group();
  millRotor.position.set(-40, -5.3 + mill.position.y + 2.8, 9.04);
  scene.add(millRotor);
  const sailRoot = new THREE.Group();
  ball(sailRoot, wood, 0, 0, 0, .22, .22, .16, true);
  for (let i = 0; i < 4; i++) {
    const blade = new THREE.Group();
    blade.rotation.z = i * Math.PI / 2;
    sailRoot.add(blade);
    box(blade, wood, 0, 1.05, 0, .075, 2.25, .065);
    box(blade, cream, .22, 1.28, -.018, .52, 1.5, .055);
    for (let j = 0; j < 5; j++) box(blade, wood, .23, .65 + j * .31, .022, .54, .04, .04);
  }
  millRotor.add(mergeStatic(sailRoot));
  // Curving wheat rows give the little mill an unmistakable silhouette from above.
  const wheat = material("#e5be64", { roughness: 1 });
  for (let row = 0; row < 6; row++) for (let i = 0; i < 15; i++) {
    const x = -4.5 + i * .26, z = -.8 + row * .49, y = windmillIsle.height(x, z);
    cylinder(windmillIsle.root, wheat, .065, .018, .32, x, y + .16, z, 5);
  }
  for (let i = 0; i < 11; i++) {
    const angle = i / 11 * Math.PI * 2, x = Math.cos(angle) * 5.2, z = Math.sin(angle) * 5.2;
    if (z > .5 && x < -1) continue;
    const y = windmillIsle.height(x, z);
    cylinder(windmillIsle.root, trunk, .08, .13, 1.1, x, y + .55, z, 7);
    ball(windmillIsle.root, i % 3 ? leaf : leafLight, x, y + 1.4, z, .62, .9, .64, true);
  }
  const gardenIsle = littleIsland(-19, 29, 5.9, 2.8);
  // A stone moon gate and a cherry grove are the reward at the southern turn.
  const gate = new THREE.Group();
  gate.position.set(0, gardenIsle.height(0, 0), 0);
  gate.rotation.y = -.4;
  gardenIsle.root.add(gate);
  const moonGate = new THREE.Mesh(new THREE.TorusGeometry(1.52, .19, 8, 56), cream);
  moonGate.position.y = 1.58;
  gate.add(moonGate);
  box(gate, rock, 0, .09, 0, 3.65, .18, .88);
  for (const x of [-1.9, 1.9]) {
    box(gate, rock, x, .17, 0, .5, .35, .5);
    cylinder(gate, cream, .18, .26, .53, x, .62, 0, 6);
    ball(gate, gardenGlow, x, .97, 0, .19, .19, .19, true);
    cylinder(gate, blueRoof, .03, .33, .23, x, 1.21, 0, 6);
  }
  for (let i = 0; i < 9; i++) {
    const angle = i / 9 * Math.PI * 2, x = Math.cos(angle) * 3.6, z = Math.sin(angle) * 3.6;
    if (z > 1 && Math.abs(x) < 2) continue;
    const y = gardenIsle.height(x, z), size = .85 + (i % 3) * .13;
    cylinder(gardenIsle.root, trunk, .07, .15, 1.55 * size, x, y + .7 * size, z, 7);
    for (const [dx, dy, dz, r] of [[0, 1.95, 0, .95], [-.65, 1.57, .08, .65], [.62, 1.6, -.05, .67]])
      ball(gardenIsle.root, i % 2 ? blossom : blossomLight, x + dx * size, y + dy * size, z + dz * size, r * size, r * .69 * size, r * .85 * size, true);
  }
  for (let i = 0; i < 12; i++) {
    const z = 1 + i * .3, x = Math.sin(z * .7) * .3;
    ball(gardenIsle.root, cream, x, gardenIsle.height(x, z) + .06, z, .27, .055, .19, true);
  }
  for (let i = 0; i < 55; i++) {
    const angle = range(0, Math.PI * 2), r = range(2.7, 5), x = Math.cos(angle) * r, z = Math.sin(angle) * r;
    ball(gardenIsle.root, i % 3 ? lavender : gardenGlow, x, gardenIsle.height(x, z) + .12, z, .09, .13, .09, true);
  }
  // Sea stacks and a quiet third island pull the eye out toward the horizon.
  const stackIsle = littleIsland(33, -32, 4.6, 1.2);
  for (const [x, z, h, radius] of [[-.8, -.6, 5, 1.25], [1.3, .7, 3.8, .93], [-2.2, .3, 2.4, .7]]) {
    const y = stackIsle.height(x, z);
    cylinder(stackIsle.root, rock, radius * .57, radius, h, x, y + h * .5, z, 7);
    ball(stackIsle.root, grass, x, y + h, z, radius * .62, .18, radius * .61, true);
  }
  const distantMat = new THREE.MeshBasicMaterial({ color: "#83b5cb", fog: true });
  for (const [x, z, sx, height, sz] of [[-92, -108, 19, 13, 11], [-67, -117, 16, 8, 10], [28, -138, 25, 16, 12], [55, -126, 19, 11, 13], [94, -96, 23, 12, 14], [-118, 47, 19, 9, 14], [81, 104, 23, 11, 14]]) {
    ball(remoteRoot, distantMat, x, -5.3 - height * .28, z, sx, height, sz, true);
    ball(remoteRoot, distantMat, x + sx * .68, -5.3 - height * .15, z + 2, sx * .58, height * .61, sz * .7, true);
  }
  scene.add(mergeStatic(remoteRoot));

  // Stitched hot-air balloons drift over the course: one material per envelope.
  const balloons = [];
  function hotAirBalloon(x, y, z, scale, colors) {
    const root = new THREE.Group();
    root.position.set(x, y, z);
    root.scale.setScalar(scale);
    scene.add(root);
    const construction = new THREE.Group(), vertices = [], colorData = [], indices = [];
    const palette = colors.map((c) => new THREE.Color(c));
    const columns = 48, rows = 24;
    for (let j = 0; j <= rows; j++) for (let i = 0; i <= columns; i++) {
      const v = j / rows, theta = v * Math.PI, angle = i / columns * Math.PI * 2;
      const radius = Math.sin(theta) * (1.8 + .48 * Math.cos(theta));
      vertices.push(Math.cos(angle) * radius, 3.65 + Math.cos(theta) * 2.35, Math.sin(angle) * radius);
      const color = palette[Math.floor((i % columns) / 4) % palette.length];
      colorData.push(color.r, color.g, color.b);
    }
    for (let j = 0; j < rows; j++) for (let i = 0; i < columns; i++) {
      const a = j * (columns + 1) + i;
      indices.push(a, a + 1, a + columns + 1, a + 1, a + columns + 2, a + columns + 1);
    }
    const envelopeGeo = new THREE.BufferGeometry();
    envelopeGeo.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    envelopeGeo.setAttribute("color", new THREE.Float32BufferAttribute(colorData, 3));
    envelopeGeo.setIndex(indices);
    envelopeGeo.computeVertexNormals();
    const envelope = new THREE.Mesh(envelopeGeo, material("#ffffff", { roughness: .85, vertexColors: true }));
    envelope.castShadow = true;
    root.add(envelope);
    box(construction, wood, 0, .22, 0, .76, .53, .61);
    box(construction, cream, 0, .49, 0, .8, .085, .64);
    for (const px of [-.31, .31]) for (const pz of [-.24, .24]) {
      const rope = cylinder(construction, wood, .016, .016, 1.21, px, 1.08, pz, 5);
      rope.rotation.z = -px * .2;
    }
    cylinder(construction, wood, .35, .29, .15, 0, 1.46, 0, 16);
    root.add(mergeStatic(construction));
    balloons.push({ root, x, y, z });
  }
  hotAirBalloon(21, 11, 36, 1.2, ["#efab73", "#fff0cf", "#de7d75", "#fff0cf"]);
  hotAirBalloon(-62, 8, -50, 1.45, ["#70b9c2", "#fff0cf", "#72a7c4", "#fff0cf"]);
  hotAirBalloon(45, 23, -59, 1.25, ["#ceb0d8", "#fff0cf", "#ecbd85", "#fff0cf"]);

  const landmarks = [
    { id: "village", name: "Himitsu Village", position: new THREE.Vector3(-4, 1, -12), radius: 19, description: "Little red roofs, a seaside lighthouse, and a familiar pink door." },
    { id: "windmill", name: "Windmill Cay", position: new THREE.Vector3(-40, 4, 8), radius: 19, description: "Follow the turning sails above fields of golden wheat." },
    { id: "garden", name: "Moonflower Garden", position: new THREE.Vector3(-19, 1, 29), radius: 19, description: "A round moon gate rests among the cherry blossoms." },
    { id: "balloons", name: "Balloon Crossing", position: new THREE.Vector3(12, 16, 29), radius: 20, description: "Striped balloons carry quiet wishes on the ocean breeze." },
  ];

  // Clouds are instanced: hundreds of soft lobes in a single draw call.
  const cloudMat = material("#fff9eb", {
    roughness: 1,
    transparent: true,
    depthWrite: false,
    emissive: "#bad9ef",
    emissiveIntensity: 0.12,
  });
  cloudMat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace(
        "#include <common>",
        "#include <common>\nattribute float instanceOpacity;\nvarying float vCloudOpacity;",
      )
      .replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\nvCloudOpacity=instanceOpacity;",
      );
    shader.fragmentShader = shader.fragmentShader
      .replace(
        "#include <common>",
        "#include <common>\nvarying float vCloudOpacity;",
      )
      .replace(
        "#include <alphatest_fragment>",
        "diffuseColor.a *= vCloudOpacity;\n#include <alphatest_fragment>",
      );
  };
  cloudMat.customProgramCacheKey = () => "pocket-skies-cloud-visibility";
  const cloudGeo = new THREE.SphereGeometry(1, 16, 12);
  const cloudData = [];
  const cloudLobes = [
    [0, 0.15, 0, 1.4, 1.0],
    [-1.5, -0.25, 0, 1.2, 0.65],
    [1.5, -0.2, 0, 1.2, 0.68],
    [-0.55, 0.63, -0.2, 1.0, 1.0],
    [0.85, 0.58, -0.25, 1.0, 0.95],
    [2.5, -0.3, 0.1, 0.85, 0.55],
  ];
  for (let i = 0; i < 25; i++) {
    const a = (i / 25) * Math.PI * 2,
      r = range(43, 135),
      cx = Math.cos(a) * r,
      cz = Math.sin(a) * r,
      cy = range(9, 37),
      scale = range(1.4, 3.1);
    for (const [x, y, z, sx, sy] of cloudLobes)
      cloudData.push({
        x: cx + x * scale,
        y: cy + y * scale,
        z: cz + z * scale,
        sx: scale * sx,
        sy: scale * sy,
        sz: scale * range(0.8, 1.1),
      });
  }
  for (const [cx, cy, cz, scale] of [
    [-16, 11, -15, 1.8],
    [8, 15, -25, 2.4],
    [22, 9, -12, 1.9],
    [-24, 16, -26, 2.3],
  ])
    for (const [x, y, z, sx, sy] of cloudLobes)
      cloudData.push({
        x: cx + x * scale,
        y: cy + y * scale,
        z: cz + z * scale,
        sx: scale * sx,
        sy: scale * sy,
        sz: scale * 0.9,
      });
  const cloudOpacity = new THREE.InstancedBufferAttribute(
    new Float32Array(cloudData.length).fill(1),
    1,
  );
  cloudOpacity.setUsage(THREE.DynamicDrawUsage);
  cloudGeo.setAttribute("instanceOpacity", cloudOpacity);
  const sightDirection = new THREE.Vector3(),
    sightClosest = new THREE.Vector3(),
    sightRelative = new THREE.Vector3(),
    subjectCenter = new THREE.Vector3();
  const clouds = new THREE.InstancedMesh(cloudGeo, cloudMat, cloudData.length);
  clouds.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(clouds);
  const dummy = new THREE.Object3D();
  function moveClouds(t, dt = 0, camera = null, subject = null) {
    let sightLength = 1;
    if (camera && subject) {
      subjectCenter.copy(subject).y += 1.6;
      sightDirection.copy(subjectCenter).sub(camera.position);
      sightLength = Math.max(0.1, sightDirection.lengthSq());
    }
    cloudData.forEach((c, i) => {
      dummy.position.set(c.x + Math.sin(t * 0.015 + i * 0.1) * 1.3, c.y, c.z);
      if (camera && subject) {
        const size = Math.max(c.sx, c.sy, c.sz);
        const nearby = THREE.MathUtils.smoothstep(
          dummy.position.distanceTo(subjectCenter),
          4 + size,
          8 + size,
        );
        const nearCamera = THREE.MathUtils.smoothstep(
          dummy.position.distanceTo(camera.position),
          2 + size,
          5 + size,
        );
        sightRelative.copy(dummy.position).sub(camera.position);
        const fraction = sightRelative.dot(sightDirection) / sightLength;
        let clearSight = 1;
        if (fraction > 0 && fraction < 1) {
          sightClosest
            .copy(sightDirection)
            .multiplyScalar(fraction)
            .add(camera.position);
          clearSight = THREE.MathUtils.smoothstep(
            dummy.position.distanceTo(sightClosest),
            2 + size,
            5 + size,
          );
        }
        const opacity =
          0.015 + 0.985 * Math.min(nearby, nearCamera, clearSight);
        cloudOpacity.setX(
          i,
          THREE.MathUtils.lerp(
            cloudOpacity.getX(i),
            opacity,
            1 - Math.exp(-dt * 4),
          ),
        );
      }
      dummy.scale.set(c.sx, c.sy, c.sz);
      dummy.updateMatrix();
      clouds.setMatrixAt(i, dummy.matrix);
    });
    clouds.instanceMatrix.needsUpdate = true;
    cloudOpacity.needsUpdate = true;
  }
  moveClouds(0);
  const birds = [];
  const birdMat = material("#fff4df", { roughness: 1 });
  for (let i = 0; i < 9; i++) {
    const b = new THREE.Group();
    const wings = [];
    for (const side of [-1, 1]) {
      const wing = tube(
        b,
        [
          [0, 0, 0],
          [side * 0.3, 0.13, -0.02],
          [side * 0.59, 0.04, -0.08],
        ],
        0.028,
        birdMat,
      );
      wings.push(wing);
    }
    scene.add(b);
    birds.push({
      root: b,
      wings,
      angle: range(0, Math.PI * 2),
      r: range(17, 31),
      y: range(2, 11),
      speed: range(0.04, 0.1),
    });
  }
  const starsGeo = new THREE.BufferGeometry();
  const starPos = [];
  for (let i = 0; i < 110; i++)
    starPos.push(range(-65, 65), range(0, 30), range(-65, 65));
  starsGeo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(starPos, 3),
  );
  const motes = new THREE.Points(
    starsGeo,
    new THREE.PointsMaterial({
      color: "#fff3be",
      size: 0.065,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    }),
  );
  motes.material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace("#include <alphatest_fragment>", `
      float moteRadius = length(gl_PointCoord - .5);
      diffuseColor.a *= 1. - smoothstep(.15, .5, moteRadius);
      #include <alphatest_fragment>
    `);
  };
  scene.add(motes);
  const palettes = {
    day: {
      top: "#258dca",
      bottom: "#a9d8ef",
      water: "#42acbc",
      shine: "#daf7ed",
      sun: "#fff5cf",
      cloud: "#fff9eb",
      fog: "#aed5e9",
      light: "#fff3d5",
      hemi: "#b6e0ef",
      ground: "#8da7b3",
      night: 0,
    },
    sunset: {
      top: "#8c98ce",
      bottom: "#f6d3ab",
      water: "#7fa5b8",
      shine: "#ffdfb1",
      sun: "#ffdb9c",
      cloud: "#ffe6c7",
      fog: "#e7c6b5",
      light: "#ffd09c",
      hemi: "#bcc4ef",
      ground: "#aa8085",
      night: 0,
    },
    night: {
      top: "#10253f",
      bottom: "#3b597b",
      water: "#294e69",
      shine: "#93c9cb",
      sun: "#e5f1d5",
      cloud: "#a2bcca",
      fog: "#405e74",
      light: "#c9dfef",
      hemi: "#8daacc",
      ground: "#525874",
      night: 1,
    },
  };
  const paletteColors = Object.fromEntries(Object.entries(palettes).map(([name, palette]) => [
    name, Object.fromEntries(Object.entries(palette).filter(([key]) => key !== "night").map(([key, color]) => [key, new THREE.Color(color)])),
  ]));
  let current = "day";
  let cloudTick = 0;
  return {
    groundHeight,
    landmarks,
    flightFloor(x, z) {
      const mainFloor = -1.8 +
        5.6 *
          (1 -
            THREE.MathUtils.smoothstep(
              Math.hypot(x - diorama.position.x, z - diorama.position.z),
              9,
              14,
            ));
      let floor = mainFloor;
      for (const [cx, cz, radius, height] of [[-40, 8, 7.2, 6.1], [-19, 29, 5.9, 3.8], [33, -32, 4.6, 4.4]]) {
        const distance = Math.hypot(x - cx, z - cz);
        floor = Math.max(floor, -1.8 + (height + 1.8) * (1 - THREE.MathUtils.smoothstep(distance, radius * .72, radius + 3)));
      }
      return floor;
    },
    get fadedClouds() {
      let count = 0;
      for (const opacity of cloudOpacity.array) if (opacity < 0.5) count++;
      return count;
    },
    palettes,
    sky,
    sea,
    island,
    setMood(name) {
      if (palettes[name]) current = name;
    },
    update(time, dt, camera, lights, subject) {
      const p = palettes[current], colors = paletteColors[current],
        a = 1 - Math.exp(-dt * 1.5);
      windowMat.emissive.set("#ffd38c");
      windowMat.emissiveIntensity = THREE.MathUtils.lerp(
        windowMat.emissiveIntensity,
        p.night ? 1.5 : 0,
        a,
      );
      beacon.rotation.y = time * 0.25;
      beamMat.uniforms.strength.value = THREE.MathUtils.lerp(
        beamMat.uniforms.strength.value,
        p.night ? 0.1 : 0,
        a,
      );
      lamp.intensity = THREE.MathUtils.lerp(lamp.intensity, p.night ? 7 : 0, a);
      for (const key of ["top", "bottom"])
        skyUniforms[key].value.lerp(colors[key], a);
      skyUniforms.sunColor.value.lerp(colors.sun, a);
      skyUniforms.night.value = THREE.MathUtils.lerp(
        skyUniforms.night.value,
        p.night,
        a,
      );
      distantMat.color.copy(skyUniforms.bottom.value).multiplyScalar(THREE.MathUtils.lerp(.72, .38, skyUniforms.night.value));
      skyUniforms.time.value = time;
      sky.position.copy(camera.position);
      seaUniforms.time.value = time;
      seaUniforms.night.value = skyUniforms.night.value;
      cloudMat.emissiveIntensity = THREE.MathUtils.lerp(cloudMat.emissiveIntensity, p.night ? .035 : .12, a);
      seaUniforms.water.value.lerp(colors.water, a);
      seaUniforms.horizon.value.lerp(colors.bottom, a);
      seaUniforms.shine.value.lerp(colors.shine, a);
      cloudMat.color.lerp(colors.cloud, a);
      scene.fog.color.lerp(colors.fog, a);
      lights.sun.color.lerp(colors.light, a);
      lights.hemi.color.lerp(colors.hemi, a);
      lights.hemi.groundColor.lerp(colors.ground, a);
      lights.sun.intensity = THREE.MathUtils.lerp(
        lights.sun.intensity,
        p.night ? 1.3 : 2.5,
        a,
      );
      lights.hemi.intensity = THREE.MathUtils.lerp(
        lights.hemi.intensity,
        p.night ? 1.65 : 2.2,
        a,
      );
      millRotor.rotation.z = time * .21;
      balloons.forEach((b, i) => {
        b.root.position.set(b.x + Math.sin(time * .055 + i * 2) * .8, b.y + Math.sin(time * .23 + i * 1.8) * .35, b.z);
        b.root.rotation.z = Math.sin(time * .19 + i) * .025;
      });
      gardenGlow.emissiveIntensity = THREE.MathUtils.lerp(gardenGlow.emissiveIntensity, p.night ? 1.8 : 0, a);
      boat.rotation.z = Math.sin(time * 0.8) * 0.065;
      boat.position.y = -5.05 + Math.sin(time * 0.9) * 0.075;
      if (++cloudTick % 3 === 0) moveClouds(time, dt * 3, camera, subject);
      birds.forEach((b, i) => {
        const angle = b.angle + time * b.speed;
        b.root.position.set(
          Math.cos(angle) * b.r,
          b.y + Math.sin(time * 0.7 + i) * 0.5,
          Math.sin(angle) * b.r,
        );
        b.root.rotation.y = -angle;
        b.wings.forEach(
          (w, j) =>
            (w.rotation.z = (j === 0 ? -1 : 1) * Math.sin(time * 4 + i) * 0.23),
        );
      });
      motes.rotation.y = time * 0.007;
      motes.material.opacity = THREE.MathUtils.lerp(
        motes.material.opacity,
        p.night ? 0.85 : 0.35,
        a,
      );
    },
  };
}
function goldMat() {
  return material("#dcb75e", { metalness: 0.3, roughness: 0.4 });
}
