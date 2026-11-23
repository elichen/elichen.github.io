// Dual-pass Three.js renderer for the OGBench UR5e + Robotiq 2F-85 cube scene:
//   1) Interactive 3D studio scene (orbit camera, soft shadows, workspace markings,
//      live 3D world-model imagined cube trajectories, uncertainty halo, belief marker).
//   2) Exact offscreen 224x224 `front_pixels` MuJoCo-matched render pass fed directly
//      into ViT-tiny -> Readout2 -> LatentSkip1V2 (zero privileged state!).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { WORKSPACE_LO, WORKSPACE_HI } from './cube-env.js';

const EYE_SIZE = 224;
// Exact OGBench `front_pixels` camera pose & vertical FOV
const EYE_POS = [1.053, -0.014, 0.639];
const EYE_QUAT_WXYZ = [0.63798595, 0.30491627, 0.30491627, 0.63798595];
const EYE_FOVY = 45.0;

function cssColor(name, fallback) {
  const v = typeof getComputedStyle === 'function'
    ? getComputedStyle(document.documentElement).getPropertyValue(name).trim()
    : '';
  return new THREE.Color(v || fallback);
}

// Build exact MuJoCo builtin="checker" floor texture straight from env.model.tex_data (tex 0: 512x512 RGB).
// In MuJoCo, texuniform="true" with texrepeat="1 1" spans 2.0m x 2.0m per full 2x2 checker tile (1m x 1m squares),
// with the 1-pixel cross mark lying directly on x = 0 and y = 0.
function makeMujocoGridTexture(model) {
  const S = model.tex_width[0] || 512;
  const adr = model.tex_adr[0] || 0;
  const src = model.tex_data;
  const rgba = new Uint8Array(S * S * 4);
  for (let i = 0, j = 0; i < S * S * 3; i += 3, j += 4) {
    rgba[j] = src[adr + i];
    rgba[j + 1] = src[adr + i + 1];
    rgba[j + 2] = src[adr + i + 2];
    rgba[j + 3] = 255;
  }
  const tex = new THREE.DataTexture(rgba, S, S, THREE.RGBAFormat);
  tex.flipY = true;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);      // PlaneGeometry(4, 4) / 2 = 2.0m per full 2x2 checker tile
  tex.offset.set(0.5, 0.5);  // align tile center (x=256, y=256) onto world (x=0, y=0)
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.colorSpace = THREE.NoColorSpace;
  tex.anisotropy = 8;
  tex.needsUpdate = true;
  return tex;
}

// Studio floor texture: subtle drafting grid on warm slate-sand surface
function makeStudioTableTexture() {
  const S = 512;
  const c = document.createElement('canvas');
  c.width = S; c.height = S;
  const g = c.getContext('2d');
  g.fillStyle = '#252c38';
  g.fillRect(0, 0, S, S);
  g.fillStyle = '#2a3240';
  g.fillRect(0, 0, S / 2, S / 2);
  g.fillRect(S / 2, S / 2, S / 2, S / 2);
  g.strokeStyle = 'rgba(218, 224, 234, 0.08)';
  g.lineWidth = 2;
  g.strokeRect(0, 0, S, S);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(8, 8);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

export class SceneRenderer {
  constructor(canvas, env) {
    this.canvas = canvas;
    this.env = env;
    const m = env.model;

    const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;

    // Shared geometries built once from MuJoCo's loaded mesh buffers
    const geometries = {};
    const getMeshGeo = (id) => {
      if (!geometries[id]) {
        const v0 = m.mesh_vertadr[id], nv = m.mesh_vertnum[id];
        const f0 = m.mesh_faceadr[id], nf = m.mesh_facenum[id];
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), 3));
        geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))), 1));
        geometries[id] = toCreasedNormals(geo, Math.PI / 5);
      }
      return geometries[id];
    };

    // -----------------------------------------------------------------
    // 1. Studio Scene (interactive 3D viewport)
    // -----------------------------------------------------------------
    const scene = this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.background = cssColor('--viewport-bg', '#171c26');
    scene.fog = new THREE.Fog(scene.background, 2.8, 7.5);

    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 30);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(1.98, -1.28, 1.15);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0.26, 0.0, 0.34);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.65;
    this.controls.maxDistance = 4.5;
    this.controls.maxPolarAngle = Math.PI * 0.485;
    this.controls.update();

    scene.add(new THREE.HemisphereLight(0xeef3ff, 0x2a303c, 0.55));
    const keyLight = new THREE.DirectionalLight(0xfffaf2, 1.5);
    keyLight.position.set(0.9, -0.7, 1.9);
    keyLight.target.position.set(0.4, 0, 0);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(2048, 2048);
    Object.assign(keyLight.shadow.camera, { left: -1.1, right: 1.1, top: 1.1, bottom: -1.1, near: 0.3, far: 5 });
    keyLight.shadow.bias = -0.0004;
    keyLight.shadow.normalBias = 0.008;
    scene.add(keyLight, keyLight.target);

    const fillLight = new THREE.DirectionalLight(0xc8dcff, 0.45);
    fillLight.position.set(-0.6, 0.9, 1.1);
    scene.add(fillLight);

    // Table surface + workspace boundary outline
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      new THREE.MeshStandardMaterial({ map: makeStudioTableTexture(), roughness: 0.92, metalness: 0.05, envMapIntensity: 0.25 })
    );
    floor.receiveShadow = true;
    scene.add(floor);
    this.studioFloor = floor;

    const [x0, y0] = WORKSPACE_LO, [x1, y1] = WORKSPACE_HI;
    const wsPts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]].map(([x, y]) => new THREE.Vector3(x, y, 0.0012));
    const wsLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(wsPts),
      new THREE.LineDashedMaterial({ color: 0x9aa8bf, dashSize: 0.018, gapSize: 0.012, transparent: true, opacity: 0.45 })
    );
    wsLine.computeLineDistances();
    scene.add(wsLine);

    // Build studio robot meshes (group 2 visual meshes)
    const studioMats = {};
    this.studioParts = [];
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_group[g] !== 2 || m.geom_type[g] !== 7) continue;
      const geo = getMeshGeo(m.geom_dataid[g]);
      const matId = m.geom_matid[g];
      if (!studioMats[matId]) {
        const c = m.mat_rgba.slice(4 * matId, 4 * matId + 3);
        const isMetal = c[0] > 0.7 && c[1] > 0.7 && c[2] > 0.7;
        studioMats[matId] = new THREE.MeshStandardMaterial({
          color: new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace),
          roughness: isMetal ? 0.32 : 0.48,
          metalness: isMetal ? 0.45 : 0.12,
        });
      }
      const mesh = new THREE.Mesh(geo, studioMats[matId]);
      mesh.castShadow = mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      scene.add(mesh);
      this.studioParts.push({ geom: g, mesh });
    }

    // Cube + Goal marker in studio scene
    this.cubeGeom = [...Array(m.ngeom).keys()].find(g => m.geom_bodyid[g] === env.cubeBody);
    this.targetGeom = [...Array(m.ngeom).keys()].find(g => m.geom_bodyid[g] === env.targetBody);
    const cs = m.geom_size.slice(3 * this.cubeGeom, 3 * this.cubeGeom + 3);
    const cubeGeo = new THREE.BoxGeometry(2 * cs[0], 2 * cs[1], 2 * cs[2]);

    this.cubeMesh = new THREE.Mesh(cubeGeo, new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(0.96, 0.26, 0.33, THREE.SRGBColorSpace),
      roughness: 0.42, metalness: 0.04,
    }));
    this.cubeMesh.castShadow = this.cubeMesh.receiveShadow = true;
    this.cubeMesh.matrixAutoUpdate = false;
    scene.add(this.cubeMesh);

    // Crisp translucent goal box + glowing table ring + vertical drop-line for lifted goals
    this.goalGroup = new THREE.Group();
    const goalBox = new THREE.Mesh(cubeGeo, new THREE.MeshStandardMaterial({
      color: new THREE.Color().setRGB(0.96, 0.26, 0.33, THREE.SRGBColorSpace),
      transparent: true, opacity: 0.24, roughness: 0.6, depthWrite: false,
    }));
    const goalEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(cubeGeo),
      new THREE.LineBasicMaterial({ color: 0xff6b78, transparent: true, opacity: 0.85 })
    );
    this.goalRing = new THREE.Mesh(
      new THREE.RingGeometry(0.036, 0.041, 48),
      new THREE.MeshBasicMaterial({ color: 0xff6b78, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false })
    );
    this.goalRing.position.z = -0.0185;
    const dropGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, -0.1)]);
    this.goalDropLine = new THREE.Line(
      dropGeo,
      new THREE.LineDashedMaterial({ color: 0xff6b78, dashSize: 0.01, gapSize: 0.008, transparent: true, opacity: 0.7 })
    );
    this.goalDropLine.visible = false;
    this.goalGroup.add(goalBox, goalEdges, this.goalRing, this.goalDropLine);
    this.showGoalMarker = true;
    scene.add(this.goalGroup);

    // World-model imagination overlay (3D imagined cube trajectories + uncertainty halo + Readout2 belief)
    this.imaginationGroup = new THREE.Group();
    scene.add(this.imaginationGroup);
    this.showImagination = true;

    this.beliefMarker = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(0.044, 0.044, 0.044)),
      new THREE.LineBasicMaterial({ color: 0x56d4c8, transparent: true, opacity: 0.85 })
    );
    this.beliefMarker.visible = false;
    this.imaginationGroup.add(this.beliefMarker);

    // Pool of 12 candidate trajectory lines + breadcrumb spheres along best imagined cube path + uncertainty halo
    this.candLines = [];
    for (let i = 0; i < 12; i++) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(11 * 3), 3));
      const mat = new THREE.LineBasicMaterial({
        color: i === 0 ? 0x56d4c8 : 0x7cb8ff,
        transparent: true,
        opacity: i === 0 ? 0.95 : 0.32,
        depthWrite: false,
      });
      const line = new THREE.Line(geo, mat);
      line.visible = false;
      this.imaginationGroup.add(line);
      this.candLines.push(line);
    }
    this.bestBeads = [];
    const beadGeo = new THREE.SphereGeometry(0.0045, 10, 10);
    for (let t = 0; t < 10; t++) {
      const bead = new THREE.Mesh(
        beadGeo,
        new THREE.MeshBasicMaterial({ color: 0x56d4c8, transparent: true, opacity: 0.35 + 0.06 * t, depthWrite: false })
      );
      bead.visible = false;
      this.imaginationGroup.add(bead);
      this.bestBeads.push(bead);
    }
    this.bestEndDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.0075, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x56d4c8, transparent: true, opacity: 0.95 })
    );
    this.bestEndDot.visible = false;
    this.imaginationGroup.add(this.bestEndDot);

    this.uncHalo = new THREE.Mesh(
      new THREE.SphereGeometry(1, 20, 16),
      new THREE.MeshBasicMaterial({ color: 0x56d4c8, transparent: true, opacity: 0.14, wireframe: true, depthWrite: false })
    );
    this.uncHalo.visible = false;
    this.imaginationGroup.add(this.uncHalo);

    // Motion trail behind the cube (lights up cleanly during throws, lifts, and pushes!)
    this.trailMax = 40;
    this.trailPts = [];
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.trailMax * 3), 3));
    this.cubeTrail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color: 0xff6b78, transparent: true, opacity: 0.78 })
    );
    this.cubeTrail.visible = false;
    scene.add(this.cubeTrail);

    // -----------------------------------------------------------------
    // 2. MuJoCo-matched Offscreen Eye Scene (`front_pixels`, 224x224)
    // -----------------------------------------------------------------
    this.eyeScene = new THREE.Scene();
    // Matches MuJoCo skybox horizon color (0.15, 0.18, 0.25) -> dark sky
    this.eyeScene.background = new THREE.Color(0.10, 0.12, 0.17);

    this.eyeCam = new THREE.PerspectiveCamera(EYE_FOVY, 1.0, 0.05, 10.0);
    this.eyeCam.position.set(EYE_POS[0], EYE_POS[1], EYE_POS[2]);
    this.eyeCam.quaternion.set(EYE_QUAT_WXYZ[1], EYE_QUAT_WXYZ[2], EYE_QUAT_WXYZ[3], EYE_QUAT_WXYZ[0]);
    this.eyeCam.updateMatrixWorld(true);

    // Exact floor_wall.xml lighting in classic OpenGL (naive sRGB space).
    // Three.js r155+ physical lights divide diffuse irradiance by Math.PI, so multiply by PI
    // to match MuJoCo's classic Blinn-Phong intensity:
    //   headlight: ambient 0.1, diffuse 0.6 along camera viewing direction
    //   global light: pos=(0,0,2) directional, diffuse 0.7, castShadow=true
    //   spotlight: pos=(0.25,0,0.5) dir=(0,0,-1), diffuse 0.3, castShadow=false
    const PI = Math.PI;
    // Exact floor_wall.xml lighting (Three.js r155+ MeshPhongMaterial divides ALL light irradiance by PI):
    //   headlight: ambient 0.10, diffuse 0.60 along camera viewing direction (0.778, 0, 0.628)
    //   global light: pos=(0,0,2) directional, diffuse 0.70, castShadow=true
    //   spotlight: pos=(0.25,0,0.5) cutoff=100° -> PointLight(decay=0), diffuse 0.30
    //   (plus tiny +0.02 ambient matching MuJoCo floor specular/fill so botFloor hits [41.5,52.5,73.7])
    this.eyeScene.add(new THREE.AmbientLight(0xffffff, 0.08 * PI));

    const eyeHead = new THREE.DirectionalLight(0xffffff, 0.60 * PI);
    eyeHead.position.set(0.778, 0.0, 0.628);
    eyeHead.target.position.set(0, 0, 0);
    this.eyeScene.add(eyeHead, eyeHead.target);

    const eyeGlobal = new THREE.DirectionalLight(0xffffff, 0.74 * PI);
    eyeGlobal.position.set(0, 0, 2);
    eyeGlobal.target.position.set(0, 0, 0);
    eyeGlobal.castShadow = true;
    eyeGlobal.shadow.mapSize.set(1024, 1024);
    Object.assign(eyeGlobal.shadow.camera, { left: -0.9, right: 0.9, top: 0.9, bottom: -0.9, near: 0.5, far: 3.5 });
    eyeGlobal.shadow.bias = -0.001;
    this.eyeScene.add(eyeGlobal, eyeGlobal.target);

    const eyeSpot = new THREE.PointLight(0xffffff, 0.30 * PI, 0, 0);
    eyeSpot.position.set(0.25, 0, 0.5);
    this.eyeScene.add(eyeSpot);

    // MuJoCo checker floor (4m x 4m plane, exact tex_data from model)
    const eyeFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(4, 4),
      new THREE.MeshPhongMaterial({
        map: makeMujocoGridTexture(m),
        specular: new THREE.Color(0.0, 0.0, 0.0),
        shininess: 16,
      })
    );
    eyeFloor.receiveShadow = true;
    this.eyeScene.add(eyeFloor);

    // Robot meshes in eyeScene with raw MuJoCo mat_rgba (no sRGB linearization -> classic OpenGL parity!)
    const eyeMats = {};
    this.eyeParts = [];
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_group[g] !== 2 || m.geom_type[g] !== 7) continue;
      const geo = getMeshGeo(m.geom_dataid[g]);
      const matId = m.geom_matid[g];
      if (!eyeMats[matId]) {
        const c = m.mat_rgba.slice(4 * matId, 4 * matId + 3);
        eyeMats[matId] = new THREE.MeshPhongMaterial({
          color: new THREE.Color(c[0], c[1], c[2]),
          specular: new THREE.Color(0.10, 0.10, 0.10),
          shininess: 32,
        });
      }
      const mesh = new THREE.Mesh(geo, eyeMats[matId]);
      mesh.castShadow = mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      this.eyeScene.add(mesh);
      this.eyeParts.push({ geom: g, mesh });
    }

    this.eyeCube = new THREE.Mesh(cubeGeo, new THREE.MeshPhongMaterial({
      color: new THREE.Color(0.96, 0.26, 0.33),
      specular: new THREE.Color(0.08, 0.08, 0.08),
      shininess: 24,
    }));
    this.eyeCube.castShadow = this.eyeCube.receiveShadow = true;
    this.eyeCube.matrixAutoUpdate = false;
    this.eyeScene.add(this.eyeCube);

    this.eyeTarget = new THREE.Mesh(cubeGeo, new THREE.MeshPhongMaterial({
      color: new THREE.Color(0.96, 0.26, 0.33),
      transparent: true, opacity: 0.2, depthWrite: false,
    }));
    this.eyeTarget.matrixAutoUpdate = false;
    this.eyeScene.add(this.eyeTarget);

    this.eyeRT = new THREE.WebGLRenderTarget(EYE_SIZE, EYE_SIZE, {
      type: THREE.UnsignedByteType,
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    });
    this._rgbaBuf = new Uint8Array(EYE_SIZE * EYE_SIZE * 4);
    this._rgbFlipped = new Uint8Array(EYE_SIZE * EYE_SIZE * 3);

    this.resize();
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(() => this.resize()).observe(canvas);
    }
  }

  resize() {
    const w = this.canvas.clientWidth || this.canvas.width || 640;
    const h = this.canvas.clientHeight || this.canvas.height || 360;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  static pose(obj, p, r, i) {
    const e = obj.matrix.elements;
    e[0] = r[9 * i]; e[4] = r[9 * i + 1]; e[8] = r[9 * i + 2];
    e[1] = r[9 * i + 3]; e[5] = r[9 * i + 4]; e[9] = r[9 * i + 5];
    e[2] = r[9 * i + 6]; e[6] = r[9 * i + 7]; e[10] = r[9 * i + 8];
    e[12] = p[3 * i]; e[13] = p[3 * i + 1]; e[14] = p[3 * i + 2];
    e[3] = e[7] = e[11] = 0; e[15] = 1;
    obj.matrixWorldNeedsUpdate = true;
  }

  syncPoses() {
    const d = this.env.data;
    const xpos = d.geom_xpos, xmat = d.geom_xmat;
    for (const { geom, mesh } of this.studioParts) SceneRenderer.pose(mesh, xpos, xmat, geom);
    for (const { geom, mesh } of this.eyeParts) SceneRenderer.pose(mesh, xpos, xmat, geom);
    SceneRenderer.pose(this.cubeMesh, xpos, xmat, this.cubeGeom);
    SceneRenderer.pose(this.eyeCube, xpos, xmat, this.cubeGeom);
    SceneRenderer.pose(this.eyeTarget, xpos, xmat, this.targetGeom);
    this.goalGroup.visible = this.showGoalMarker;
    this.goalGroup.position.set(this.env.goal[0], this.env.goal[1], this.env.goal[2]);
    // Keep ring grounded on the table and show vertical tether when goal is lifted
    const gz = this.env.goal[2];
    this.goalRing.position.z = 0.0015 - gz;
    if (gz > 0.032) {
      const pos = this.goalDropLine.geometry.attributes.position;
      pos.setXYZ(0, 0, 0, -0.02);
      pos.setXYZ(1, 0, 0, 0.0015 - gz);
      pos.needsUpdate = true;
      this.goalDropLine.computeLineDistances();
      this.goalDropLine.visible = true;
    } else {
      this.goalDropLine.visible = false;
    }
  }

  // Render the exact 224x224 `front_pixels` view and return top-to-bottom RGB Uint8Array(224*224*3).
  // Optionally blits the captured image onto a 2D preview <canvas> (the "Model's eye" PIP).
  captureEyePixels(previewCanvas = null, showTargetInEye = true) {
    this.syncPoses();
    this.eyeTarget.visible = showTargetInEye;

    const r = this.renderer;
    const prevTM = r.toneMapping;
    const prevCS = r.outputColorSpace;
    const prevTarget = r.getRenderTarget();

    // Render in classic OpenGL naive-sRGB space (no ACES, no linear->sRGB double gamma)
    r.toneMapping = THREE.NoToneMapping;
    r.outputColorSpace = THREE.LinearSRGBColorSpace;
    r.setRenderTarget(this.eyeRT);
    r.clear();
    r.render(this.eyeScene, this.eyeCam);
    r.readRenderTargetPixels(this.eyeRT, 0, 0, EYE_SIZE, EYE_SIZE, this._rgbaBuf);

    r.setRenderTarget(prevTarget);
    r.toneMapping = prevTM;
    r.outputColorSpace = prevCS;

    // Flip vertically (WebGL bottom-up -> top-down RGB)
    const src = this._rgbaBuf, dst = this._rgbFlipped;
    for (let y = 0; y < EYE_SIZE; y++) {
      const sy = EYE_SIZE - 1 - y;
      let sOff = sy * EYE_SIZE * 4;
      let dOff = y * EYE_SIZE * 3;
      for (let x = 0; x < EYE_SIZE; x++) {
        dst[dOff++] = src[sOff++];
        dst[dOff++] = src[sOff++];
        dst[dOff++] = src[sOff++];
        sOff++;
      }
    }

    if (previewCanvas) {
      const ctx = previewCanvas.getContext('2d');
      if (ctx) {
        if (!this._imgData) this._imgData = ctx.createImageData(EYE_SIZE, EYE_SIZE);
        const out = this._imgData.data;
        let k = 0;
        for (let i = 0; i < dst.length; i += 3) {
          out[k++] = dst[i];
          out[k++] = dst[i + 1];
          out[k++] = dst[i + 2];
          out[k++] = 255;
        }
        ctx.putImageData(this._imgData, 0, 0);
      }
    }
    return dst;
  }

  // Update 3D visualization of the world model's imagined futures & Readout2 belief
  updateImagination(plan) {
    this.imaginationGroup.visible = this.showImagination && !!plan;
    if (!plan) return;

    // Cyan wireframe box at where Readout2 believes the cube is (purely from pixels!)
    this.beliefMarker.position.set(plan.s0[0], plan.s0[1], plan.s0[2]);
    this.beliefMarker.visible = true;

    const top = plan.topTrajs || [];
    for (let i = 0; i < this.candLines.length; i++) {
      const line = this.candLines[i];
      if (i >= top.length) { line.visible = false; continue; }
      const tr = top[i];
      const pts = tr.cube;
      const pos = line.geometry.attributes.position.array;
      // Start trajectory from the visual readout cube position s0[:3]
      pos[0] = plan.s0[0]; pos[1] = plan.s0[1]; pos[2] = plan.s0[2];
      for (let t = 0; t < pts.length; t++) {
        pos[(t + 1) * 3] = pts[t][0];
        pos[(t + 1) * 3 + 1] = pts[t][1];
        pos[(t + 1) * 3 + 2] = pts[t][2];
      }
      line.geometry.setDrawRange(0, pts.length + 1);
      line.geometry.attributes.position.needsUpdate = true;
      line.visible = true;
    }

    const best = plan.bestTraj;
    for (let t = 0; t < this.bestBeads.length; t++) {
      const b = this.bestBeads[t];
      if (best && t < best.cube.length) {
        const p = best.cube[t];
        b.position.set(p[0], p[1], p[2]);
        b.visible = true;
      } else {
        b.visible = false;
      }
    }
    if (best && best.cube.length) {
      const end = best.cube[best.cube.length - 1];
      this.bestEndDot.position.set(end[0], end[1], end[2]);
      this.bestEndDot.visible = true;
      const sig = Math.max(0.01, Math.min(0.12, plan.bestSig || 0.02));
      this.uncHalo.position.set(end[0], end[1], end[2]);
      this.uncHalo.scale.setScalar(sig);
      this.uncHalo.visible = true;
    }
  }

  // Update live visual belief box & motion trail for BehaviorPolicy mode
  updateVisualReadout(cubeEst, cubeTruePos = null) {
    if (cubeEst) {
      this.beliefMarker.position.set(cubeEst[0], cubeEst[1], cubeEst[2]);
      this.beliefMarker.visible = this.showImagination;
      this.imaginationGroup.visible = this.showImagination;
    }
    if (cubeTruePos) {
      const last = this.trailPts[this.trailPts.length - 1];
      const moved = !last || Math.hypot(cubeTruePos[0] - last[0], cubeTruePos[1] - last[1], cubeTruePos[2] - last[2]) > 0.012;
      if (moved) {
        this.trailPts.push([cubeTruePos[0], cubeTruePos[1], cubeTruePos[2]]);
        if (this.trailPts.length > this.trailMax) this.trailPts.shift();
        const pos = this.cubeTrail.geometry.attributes.position.array;
        for (let i = 0; i < this.trailPts.length; i++) {
          pos[i * 3] = this.trailPts[i][0];
          pos[i * 3 + 1] = this.trailPts[i][1];
          pos[i * 3 + 2] = this.trailPts[i][2];
        }
        this.cubeTrail.geometry.setDrawRange(0, this.trailPts.length);
        this.cubeTrail.geometry.attributes.position.needsUpdate = true;
        this.cubeTrail.visible = this.trailPts.length >= 2;
      }
    }
  }

  clearTrail() {
    this.trailPts = [];
    this.cubeTrail.visible = false;
  }

  draw() {
    this.syncPoses();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}
