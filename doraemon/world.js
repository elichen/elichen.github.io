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
  const skyUniforms = {
    top: { value: new THREE.Color("#96d4e8") },
    bottom: { value: new THREE.Color("#e4efdf") },
    sunColor: { value: new THREE.Color("#fff5cf") },
    sunDir: { value: new THREE.Vector3(-0.74, 0.025, -1).normalize() },
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
      fragmentShader:
        `varying vec3 vDir;uniform vec3 top,bottom,sunColor,sunDir;uniform float night,time;float hash(vec3 p){return fract(sin(dot(p,vec3(127.1,311.7,74.7)))*43758.5453);}void main(){vec3 d=normalize(vDir);float h=pow(max(d.y+.08,0.),.55);vec3 col=mix(bottom,top,clamp(h,0.,1.));float s=max(dot(d,sunDir),0.);col+=sunColor*pow(s,20.)*.13;col=mix(col,sunColor,smoothstep(.9984,.9990,s));vec3 cell=floor(d*310.);float star=step(.997,hash(cell))*pow(max(0.,1.-length(fract(d*310.)-.5)*2.),3.);col+=star*night*2.5*(.65+.35*sin(time+hash(cell)*30.));gl_FragColor=vec4(col,1.);#include <tonemapping_fragment>\n#include <colorspace_fragment>}`.replace(
          ";#include",
          ";\n#include",
        ),
    }),
  );
  sky.renderOrder = -10;
  scene.add(sky);
  const seaUniforms = {
    time: { value: 0 },
    water: { value: new THREE.Color("#6fbec5") },
    horizon: { value: new THREE.Color("#dcebdc") },
    shine: { value: new THREE.Color("#d3efdc") },
    night: { value: 0 },
  };
  const sea = new THREE.Mesh(
    new THREE.PlaneGeometry(1200, 1200, 1, 1),
    new THREE.ShaderMaterial({
      uniforms: seaUniforms,
      vertexShader: `varying vec3 vWorld;void main(){vec4 w=modelMatrix*vec4(position,1.);vWorld=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
      fragmentShader: `varying vec3 vWorld;uniform float time,night;uniform vec3 water,horizon,shine;void main(){vec2 p=vWorld.xz;float wave=sin(p.y*2.6+sin(p.x*.38+time*.2)*.7+time*.8);float mask=pow(max(0.,sin(p.x*.62+sin(p.y*.24))),6.);float ripple=smoothstep(.94,1.,wave)*mask*.17;float gleam=pow(max(0.,sin(p.x*.21+p.y*.41+time*.22)*sin(p.y*3.4+time*.6)),22.)*.065;vec3 col=mix(water,shine,ripple+gleam);float dist=length(cameraPosition.xz-p);col=mix(col,horizon,smoothstep(65.,260.,dist));gl_FragColor=vec4(col,1.);\n#include <tonemapping_fragment>\n#include <colorspace_fragment>}`,
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
  const sand = material("#e7d2a5", { roughness: 1 }),
    grass = material("#a3c983", { roughness: 1 }),
    rock = material("#b1b4a0", { roughness: 1 }),
    trunk = material("#957b55", { roughness: 1 }),
    leaf = material("#72a979", { roughness: 0.85 }),
    leafLight = material("#8bb880", { roughness: 1 }),
    darkLeaf = material("#518b73", { roughness: 1 });
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

  // Soft shoreline foam follows the irregular island edge.
  const foamMat = new THREE.MeshBasicMaterial({
    color: "#e4f4e5",
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
  });
  const foams = [];
  for (let k = 0; k < 3; k++) {
    const pts = [];
    for (let i = 0; i < 160; i++) {
      const a = (i / 160) * Math.PI * 2,
        r = 14.5 + k * 0.7 + 0.3 * Math.sin(a * 7);
      pts.push([Math.cos(a) * r, -5.25, Math.sin(a) * r * 0.975]);
    }
    const m = tube(diorama, pts, 0.035 + k * 0.006, foamMat, true);
    m.castShadow = false;
    foams.push(m);
  }
  // Clouds are instanced: hundreds of soft lobes in a single draw call.
  const cloudMat = material("#fff9eb", {
    roughness: 1,
    transparent: true,
    depthWrite: false,
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
  scene.add(motes);
  const palettes = {
    day: {
      top: "#96d4e8",
      bottom: "#e4efdf",
      water: "#70bdc4",
      shine: "#d3efdc",
      sun: "#fff5cf",
      cloud: "#fff9eb",
      fog: "#d6e9e3",
      light: "#fff3d5",
      hemi: "#b6e0ef",
      ground: "#8a9a7a",
      night: 0,
    },
    sunset: {
      top: "#c1b4ce",
      bottom: "#f6d3ab",
      water: "#b1b2b9",
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
      bottom: "#48677d",
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
  let current = "day";
  let cloudTick = 0;
  return {
    groundHeight,
    flightFloor(x, z) {
      return (
        -1.8 +
        5.6 *
          (1 -
            THREE.MathUtils.smoothstep(
              Math.hypot(x - diorama.position.x, z - diorama.position.z),
              9,
              14,
            ))
      );
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
      current = name;
    },
    update(time, dt, camera, lights, subject) {
      const p = palettes[current],
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
        skyUniforms[key].value.lerp(new THREE.Color(p[key]), a);
      skyUniforms.sunColor.value.lerp(new THREE.Color(p.sun), a);
      skyUniforms.night.value = THREE.MathUtils.lerp(
        skyUniforms.night.value,
        p.night,
        a,
      );
      skyUniforms.time.value = time;
      sky.position.copy(camera.position);
      seaUniforms.time.value = time;
      seaUniforms.water.value.lerp(new THREE.Color(p.water), a);
      seaUniforms.horizon.value.lerp(new THREE.Color(p.bottom), a);
      seaUniforms.shine.value.lerp(new THREE.Color(p.shine), a);
      cloudMat.color.lerp(new THREE.Color(p.cloud), a);
      scene.fog.color.lerp(new THREE.Color(p.fog), a);
      lights.sun.color.lerp(new THREE.Color(p.light), a);
      lights.hemi.color.lerp(new THREE.Color(p.hemi), a);
      lights.hemi.groundColor.lerp(new THREE.Color(p.ground), a);
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
      boat.rotation.z = Math.sin(time * 0.8) * 0.065;
      boat.position.y = -5.05 + Math.sin(time * 0.9) * 0.075;
      foams.forEach((f, i) => {
        const s = 1 + Math.sin(time * 0.6 + i) * 0.012;
        f.scale.set(s, 1, s);
      });
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
