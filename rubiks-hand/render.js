// Three.js studio renderer for Shadow Dexterous Hand + 26-cubelet physical Rubik's Cube.
// MuJoCo coordinate frame is Z-up; every body/geom pose is mirrored directly from MjData.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { FACE_INFO, CUBELET_HOME, quatRotateVec } from './cube-state.js';

// Tournament WCA sticker palette (calibrated for ACES Filmic studio lighting)
export const STICKER_HEX = {
    R: 0xd62424, // Red    (+X)
    L: 0xf26411, // Orange (-X)
    B: 0x1655c9, // Blue   (+Y)
    F: 0x149447, // Green  (-Y)
    U: 0xf6f6f0, // White  (+Z)
    D: 0xf2c216, // Yellow (-Z)
};

// Build a rounded-edge bevel cubelet geometry (18.8 mm outer, 1.5 mm chamfer) + recessed glossy tiles
function makeCubeletMesh(homeCoord, bodyMat, tileMats) {
    const group = new THREE.Group();

    // Sculpted chamfered black ABS cubelet shell using ExtrudeGeometry with bevel
    const s = 0.0076;   // flat inner half-width
    const b = 0.0017;   // bevel thickness -> total width = 2*(s+b) = 18.6 mm (crisp 0.4 mm seam gap)
    const shape = new THREE.Shape();
    shape.moveTo(-s, -s);
    shape.lineTo( s, -s);
    shape.lineTo( s,  s);
    shape.lineTo(-s,  s);
    shape.closePath();

    const geo = new THREE.ExtrudeGeometry(shape, {
        depth: 2 * s,
        bevelEnabled: true,
        bevelThickness: b,
        bevelSize: b,
        bevelSegments: 2,
    });
    geo.translate(0, 0, -s);
    const body = new THREE.Mesh(geo, bodyMat);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    // Add glossy rounded-square sticker tiles on each outward face of this cubelet
    const tileW = 0.0148, tileR = 0.0018;
    const tShape = new THREE.Shape();
    const hw = tileW * 0.5 - tileR;
    tShape.moveTo(-hw, -tileW * 0.5);
    tShape.lineTo( hw, -tileW * 0.5);
    tShape.quadraticCurveTo( tileW * 0.5, -tileW * 0.5,  tileW * 0.5, -hw);
    tShape.lineTo( tileW * 0.5,  hw);
    tShape.quadraticCurveTo( tileW * 0.5,  tileW * 0.5,  hw,  tileW * 0.5);
    tShape.lineTo(-hw,  tileW * 0.5);
    tShape.quadraticCurveTo(-tileW * 0.5,  tileW * 0.5, -tileW * 0.5,  hw);
    tShape.lineTo(-tileW * 0.5, -hw);
    tShape.quadraticCurveTo(-tileW * 0.5, -tileW * 0.5, -hw, -tileW * 0.5);
    const tileGeo = new THREE.ShapeGeometry(tShape, 4);

    const offset = 0.00945;
    const [hx, hy, hz] = homeCoord;
    const addTile = (letter, pos, rot) => {
        const m = new THREE.Mesh(tileGeo, tileMats[letter]);
        m.position.set(...pos);
        m.rotation.set(...rot);
        m.castShadow = false;
        m.receiveShadow = true;
        group.add(m);
    };
    if (hx > 0) addTile('R', [ offset, 0, 0], [0,  Math.PI / 2, 0]);
    if (hx < 0) addTile('L', [-offset, 0, 0], [0, -Math.PI / 2, 0]);
    if (hy > 0) addTile('B', [0,  offset, 0], [-Math.PI / 2, 0, 0]);
    if (hy < 0) addTile('F', [0, -offset, 0], [ Math.PI / 2, 0, 0]);
    if (hz > 0) addTile('U', [0, 0,  offset], [0, 0, 0]);
    if (hz < 0) addTile('D', [0, 0, -offset], [Math.PI, 0, 0]);

    return group;
}

export const CAMERA_PRESETS = {
    studio:  { pos: [0.58, -0.34, 0.30], target: [0.31, -0.005, 0.055] },
    closeup: { pos: [0.47, -0.22, 0.21], target: [0.32,  0.000, 0.058] },
    top:     { pos: [0.34, -0.04, 0.38], target: [0.32,  0.000, 0.050] },
    side:    { pos: [0.32, -0.38, 0.14], target: [0.30,  0.000, 0.055] },
};

export class RubikSceneRenderer {
    constructor(canvas, env) {
        this.canvas = canvas;
        this.env = env;
        const m = env.model;

        const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 0.98;

        const scene = this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        pmrem.dispose();
        scene.background = new THREE.Color('#f1ede4');
        scene.fog = new THREE.Fog('#f1ede4', 1.4, 3.8);

        this.camera = new THREE.PerspectiveCamera(32, 1, 0.01, 20);
        this.camera.up.set(0, 0, 1);
        this.camera.position.set(...CAMERA_PRESETS.studio.pos);

        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.target.set(...CAMERA_PRESETS.studio.target);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.07;
        this.controls.minDistance = 0.16;
        this.controls.maxDistance = 1.2;
        this.controls.maxPolarAngle = Math.PI * 0.49;
        this.controls.update();

        // Studio three-point lighting
        scene.add(new THREE.HemisphereLight(0xfffaf0, 0x7c7468, 0.62));
        const key = new THREE.DirectionalLight(0xfff8ee, 1.38);
        key.position.set(0.75, -0.65, 1.15);
        key.target.position.set(0.30, 0, 0.04);
        key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048);
        Object.assign(key.shadow.camera, { left: -0.45, right: 0.45, top: 0.45, bottom: -0.45, near: 0.2, far: 2.5 });
        key.shadow.bias = -0.00025;
        key.shadow.normalBias = 0.003;
        scene.add(key, key.target);

        const rim = new THREE.DirectionalLight(0xdce8ff, 0.75);
        rim.position.set(-0.4, 0.7, 0.6);
        scene.add(rim);

        // Warm matte drafting-table surface & subtle measurement ring
        const floor = new THREE.Mesh(
            new THREE.PlaneGeometry(8, 8),
            new THREE.MeshStandardMaterial({ color: 0xe6e0d4, roughness: 0.92, metalness: 0.0, envMapIntensity: 0.25 }),
        );
        floor.position.z = -0.15;
        floor.receiveShadow = true;
        scene.add(floor);

        // Subtle radial workbench mat under the hand
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x9c9486, transparent: true, opacity: 0.28, depthWrite: false });
        for (const r of [0.14, 0.24]) {
            const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.0008, r + 0.0008, 128), ringMat);
            ring.position.set(0.30, 0, -0.149);
            scene.add(ring);
        }

        // Shadow Hand visual meshes (group 2, geom_type 7 = mesh)
        const handMatMap = {
            0: new THREE.MeshStandardMaterial({ color: 0x1e2229, roughness: 0.44, metalness: 0.12, envMapIntensity: 0.65 }), // black polymer
            1: new THREE.MeshStandardMaterial({ color: 0xd0cec8, roughness: 0.38, metalness: 0.12, envMapIntensity: 0.5 }),  // forearm shell
            2: new THREE.MeshStandardMaterial({ color: 0xc5cad2, roughness: 0.28, metalness: 0.78, envMapIntensity: 0.95 }), // brushed alloy
        };
        const geometries = {};
        this.handParts = [];
        for (let g = 0; g < m.ngeom; g++) {
            if (m.geom_group[g] !== 2 || m.geom_type[g] !== 7) continue;
            const id = m.geom_dataid[g];
            if (!geometries[id]) {
                const v0 = m.mesh_vertadr[id], nv = m.mesh_vertnum[id];
                const f0 = m.mesh_faceadr[id], nf = m.mesh_facenum[id];
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), 3));
                geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))), 1));
                geometries[id] = toCreasedNormals(geo, Math.PI / 5);
            }
            const matId = m.geom_matid[g];
            const mat = handMatMap[matId] || handMatMap[0];
            const mesh = new THREE.Mesh(geometries[id], mat);
            mesh.castShadow = mesh.receiveShadow = true;
            mesh.matrixAutoUpdate = false;
            scene.add(mesh);
            this.handParts.push({ geom: g, mesh });
        }

        // 26 sculpted Rubik's cubelets (6 centers + 20 edges/corners)
        const bodyMat = new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.34, metalness: 0.06, envMapIntensity: 0.45 });
        const tileMats = {};
        for (const [k, hex] of Object.entries(STICKER_HEX)) {
            // Matte-satin speedcube plastic keeps every WCA sticker vivid below ACES desaturation knee.
            tileMats[k] = new THREE.MeshStandardMaterial({
                color: hex,
                roughness: 0.56,
                metalness: 0.0,
                envMapIntensity: 0.14,
                side: THREE.FrontSide,
            });
        }

        // Center cubelet geom names + 20 edge/corner geom names matching shadow_rubik.xml
        const cubeletDefs = [
            ...FACE_INFO.map(f => ({ geomName: `g_${f.name}`, home: f.axis })),
            ...[
                'pX_pY', 'nY_pX', 'pX_pZ', 'nZ_pX', 'nX_pY', 'nX_nY', 'nX_pZ', 'nX_nZ',
                'pY_pZ', 'nZ_pY', 'nY_pZ', 'nY_nZ',
                'pX_pY_pZ', 'nZ_pX_pY', 'nY_pX_pZ', 'nY_nZ_pX',
                'nX_pY_pZ', 'nX_nZ_pY', 'nX_nY_pZ', 'nX_nY_nZ',
            ].map((nm, i) => ({ geomName: `g_${nm}`, home: CUBELET_HOME[i] })),
        ];

        const mj = env.mj;
        this.cubelets = [];
        this.cubeHitMeshes = [];
        for (const def of cubeletDefs) {
            const g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM.value, def.geomName);
            const mesh = makeCubeletMesh(def.home, bodyMat, tileMats);
            mesh.matrixAutoUpdate = false;
            scene.add(mesh);
            this.cubelets.push({ geom: g, mesh });
            this.cubeHitMeshes.push(mesh.children[0]);
        }

        // Subtle floating target-twist halo ring above the cube during active moves
        const haloGeo = new THREE.RingGeometry(0.0325, 0.035, 64);
        this.haloMat = new THREE.MeshBasicMaterial({
            color: 0xd92b2b, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false,
        });
        this.halo = new THREE.Mesh(haloGeo, this.haloMat);
        scene.add(this.halo);

        this.camGoal = null;
        this.resize();
        new ResizeObserver(() => this.resize()).observe(canvas);
    }

    setPreset(name) {
        const p = CAMERA_PRESETS[name];
        if (!p) return;
        this.camGoal = {
            pos: new THREE.Vector3(...p.pos),
            target: new THREE.Vector3(...p.target),
            t: 0,
        };
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    static pose(obj, p, r, i) {
        const e = obj.matrix.elements;
        e[0] = r[9 * i];     e[4] = r[9 * i + 1]; e[8]  = r[9 * i + 2];
        e[1] = r[9 * i + 3]; e[5] = r[9 * i + 4]; e[9]  = r[9 * i + 5];
        e[2] = r[9 * i + 6]; e[6] = r[9 * i + 7]; e[10] = r[9 * i + 8];
        e[12] = p[3 * i];    e[13] = p[3 * i + 1]; e[14] = p[3 * i + 2];
        e[3] = e[7] = e[11] = 0; e[15] = 1;
        obj.matrixWorldNeedsUpdate = true;
    }

    draw(dt, ctrlState) {
        const d = this.env.data;
        const xpos = d.geom_xpos, xmat = d.geom_xmat;
        for (const { geom, mesh } of this.handParts) RubikSceneRenderer.pose(mesh, xpos, xmat, geom);
        for (const { geom, mesh } of this.cubelets) RubikSceneRenderer.pose(mesh, xpos, xmat, geom);

        // Update floating twist/target indicator ring above active top face
        if (ctrlState && ctrlState.phase === 'twist') {
            const cp = this.env.cubePos();
            const info = FACE_INFO[ctrlState.targetFaceId];
            this.halo.position.set(cp[0], cp[1], cp[2] + 0.036);
            this.haloMat.color.setHex(STICKER_HEX[info.letter]);
            this.haloMat.opacity = 0.50;
        } else if (ctrlState && ctrlState.phase === 'solved') {
            this.haloMat.opacity = 0.0;
        } else {
            this.haloMat.opacity = Math.max(0, this.haloMat.opacity - dt * 5.0);
        }

        // Smooth camera preset transition
        if (this.camGoal) {
            this.camGoal.t = Math.min(1, this.camGoal.t + dt * 3.5);
            const alpha = 1 - Math.pow(1 - this.camGoal.t, 3);
            this.camera.position.lerp(this.camGoal.pos, alpha * 0.25);
            this.controls.target.lerp(this.camGoal.target, alpha * 0.25);
            if (this.camGoal.t >= 1) this.camGoal = null;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
