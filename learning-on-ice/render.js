// Draws the humanoid from MuJoCo's geom poses with three.js. The floor follows
// it (snapped to whole tiles so the grid looks fixed) and blends between matte
// rubber and ice: a cold tint, skate scratches, and a mirror beneath a
// see-through surface so the body is reflected.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

const MU_ICE = 0.02, MU_RUBBER = 1.0;

function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return new THREE.Color(v || fallback);
}

// 1 m tiles: a hairline grid plus a faint diagonal texture (tread on rubber, scratches on ice)
function floorTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    g.fillStyle = '#fff';
    g.fillRect(0, 0, 256, 256);
    g.strokeStyle = 'rgba(0,0,0,0.05)';
    g.lineWidth = 2;
    for (let i = -256; i < 256; i += 16) {
        g.beginPath(); g.moveTo(i, 256); g.lineTo(i + 256, 0); g.stroke();
    }
    g.strokeStyle = 'rgba(0,0,0,0.22)';
    g.lineWidth = 3;
    g.strokeRect(0, 0, 256, 256);
    return repeating(c);
}

// Skate marks: long shallow arcs, white on transparent
function scratchTexture() {
    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const g = c.getContext('2d');
    let seed = 7;
    const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    g.lineCap = 'round';
    for (let i = 0; i < 46; i++) {
        const r = 150 + rand() * 700, a0 = rand() * Math.PI * 2, span = 0.15 + rand() * 0.35;
        const cx = rand() * 512, cy = rand() * 512;
        g.strokeStyle = `rgba(255,255,255,${0.35 + rand() * 0.5})`;
        g.lineWidth = 0.8 + rand() * 1.6;
        for (const [dx, dy] of [[0, 0], [512, 0], [-512, 0], [0, 512], [0, -512]]) {  // wrap across tile edges
            g.beginPath();
            g.arc(cx + dx, cy + dy, r, a0, a0 + span);
            g.stroke();
        }
    }
    return repeating(c);
}

function repeating(canvas) {
    const t = new THREE.CanvasTexture(canvas);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
}

export class BodyRenderer {
    constructor(canvas, info) {
        this.canvas = canvas;
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;

        const scene = this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(this.renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        this.readColors();

        // MuJoCo's world is z-up
        this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 200);
        this.camera.up.set(0, 0, 1);
        const sky = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
        sky.position.set(0, 0, 1);
        scene.add(sky);
        const sun = this.sun = new THREE.DirectionalLight(0xffffff, 2.2);
        sun.castShadow = true;
        sun.shadow.mapSize.set(1024, 1024);
        Object.assign(sun.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 20 });
        sun.shadow.bias = -0.0005;
        scene.add(sun, sun.target);

        // Floor: a mirror, the (partly see-through) surface, and skate marks on top
        // Big enough that its far edge sits on the horizon, behind the fog
        const plane = new THREE.PlaneGeometry(400, 400);
        // Untinted, so the reflected white sky stays white and the far edge of the floor doesn't show
        this.mirror = new Reflector(plane, { clipBias: 0.003, textureWidth: 512, textureHeight: 512, color: 0xffffff });
        this.mirror.position.z = -0.002;
        const tex = floorTexture();
        tex.repeat.set(400, 400);
        this.floorMat = new THREE.MeshStandardMaterial({ map: tex, transparent: true });
        this.floor = new THREE.Mesh(plane, this.floorMat);
        this.floor.receiveShadow = true;
        const scratches = scratchTexture();
        scratches.repeat.set(80, 80);
        // Tinted with the palette's emblem blue so the marks show on pale ice
        this.scratchMat = new THREE.MeshBasicMaterial({ map: scratches, color: cssColor('--text-3', '#505689'), transparent: true, depthWrite: false });
        this.scratches = new THREE.Mesh(plane, this.scratchMat);
        this.scratches.position.z = 0.002;
        this.floorGroup = new THREE.Group();
        this.floorGroup.add(this.mirror, this.floor, this.scratches);
        scene.add(this.floorGroup);

        // Body: one mesh per geom, posed from MuJoCo every frame
        this.bodyMat = new THREE.MeshStandardMaterial({ color: this.colors.body, roughness: 0.45, metalness: 0.05 });
        this.meshes = [];
        this.bandages = [];
        for (let i = 0; i < info.ngeom; i++) {
            const type = info.geomType[i], [r, half] = info.geomSize.slice(3 * i, 3 * i + 2);
            let geo = null;
            if (type === 2) geo = new THREE.SphereGeometry(r, 32, 24);
            if (type === 3) geo = new THREE.CapsuleGeometry(r, 2 * half, 6, 16).rotateX(Math.PI / 2);  // MuJoCo capsules run along local z
            if (!geo) { this.meshes.push(null); continue; }
            const mesh = new THREE.Mesh(geo, this.bodyMat);
            mesh.matrixAutoUpdate = false;
            mesh.castShadow = true;
            scene.add(mesh);
            this.meshes.push(mesh);
            // Bandage rings on the right leg, shown when it's hurt (capsules run along local z)
            if (info.geomNames[i] === 'right_shin1' || info.geomNames[i] === 'right_thigh1') {
                const bandageMat = new THREE.MeshStandardMaterial({ color: this.colors.eye, roughness: 0.8 });
                for (const z of [-0.3 * half, 0.3 * half]) {
                    const ring = new THREE.Mesh(new THREE.TorusGeometry(r + 0.004, 0.014, 8, 28), bandageMat);
                    ring.position.z = z;
                    ring.castShadow = true;
                    mesh.add(ring);
                    this.bandages.push(ring);
                }
            }
            // Eyes on the head, so you can tell which way it faces (+x is forward)
            if (info.geomNames[i] === 'head') {
                const eyeMat = new THREE.MeshStandardMaterial({ color: this.colors.eye, roughness: 0.2 });
                for (const y of [-0.035, 0.035]) {
                    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.018, 16, 12), eyeMat);
                    eye.position.set(r * 0.88, y, 0.03);
                    mesh.add(eye);
                }
            }
        }
        // Backpack, posed from the torso body's frame (x forward, z up)
        this.backpack = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.27, 0.32),
            new THREE.MeshStandardMaterial({ color: cssColor('--accent', '#3b3b61'), roughness: 0.7 }));
        this.backpack.matrixAutoUpdate = false;
        this.backpack.castShadow = true;
        this.backpackOffset = new THREE.Matrix4().makeTranslation(-0.17, 0, -0.08);
        scene.add(this.backpack);
        this.target = new THREE.Vector3();
        this.camPos = null;
        this.setWorld({ friction: MU_RUBBER, leg: 1, pack: 0 });
        this.resize();
    }

    readColors() {
        this.colors = {
            bg: cssColor('--viewport-bg', '#dfe7ec'),
            body: cssColor('--body', '#870a09'),
            eye: cssColor('--body-eye', '#ecedec'),
            ice: cssColor('--ice', '#d8ecf5'),
            iceTint: cssColor('--ice-tint', '#c8c8e7'),
            rubber: cssColor('--rubber', '#3a3d40')
        };
        this.scene.background = this.colors.bg;
        this.scene.fog = new THREE.Fog(this.colors.bg, 12, 26);
    }

    // A world from worlds.js: the floor, the bandage and the backpack
    setWorld(w) {
        this.setFriction(w.friction);
        this.bandages.forEach(b => { b.visible = w.leg < 1; });
        this.backpack.visible = w.pack > 0;
    }

    // 1 = rubber, 0 = ice, on a log scale of friction
    setFriction(mu) {
        const t = Math.min(1, Math.max(0, Math.log(mu / MU_ICE) / Math.log(MU_RUBBER / MU_ICE)));
        const ice = this.colors.ice.clone().lerp(this.colors.iceTint, 0.7);
        this.floorMat.color.copy(ice).lerp(this.colors.rubber, t);
        this.floorMat.roughness = 0.05 + 0.9 * t;
        this.floorMat.envMapIntensity = 1.0 - 0.75 * t;
        // See-through ice shows the mirror below; rubber is opaque (and skips the mirror pass)
        this.floorMat.opacity = 0.76 + 0.24 * t;
        this.mirror.visible = t < 0.95;
        this.scratchMat.opacity = 0.55 * Math.max(0, 1 - 1.6 * t);
        this.scratches.visible = this.scratchMat.opacity > 0.01;
    }

    update(frame) {
        const { xpos, xmat } = frame;
        const m = new THREE.Matrix4();
        this.meshes.forEach((mesh, i) => {
            if (!mesh) return;
            const r = xmat.subarray(9 * i, 9 * i + 9);
            m.set(r[0], r[1], r[2], xpos[3 * i],
                  r[3], r[4], r[5], xpos[3 * i + 1],
                  r[6], r[7], r[8], xpos[3 * i + 2],
                  0, 0, 0, 1);
            mesh.matrix.copy(m);
        });
        if (this.backpack.visible) {
            const r = frame.torsoMat, p = frame.torsoPos;
            m.set(r[0], r[1], r[2], p[0], r[3], r[4], r[5], p[1], r[6], r[7], r[8], p[2], 0, 0, 0, 1);
            this.backpack.matrix.multiplyMatrices(m, this.backpackOffset);
        }
        const [x, y] = frame.torso;
        this.target.set(x, y, 0.95);
    }

    render() {
        // Follow from behind and to the side, with a little lag
        const want = new THREE.Vector3(this.target.x - 2.6, this.target.y - 3.6, 1.9);
        // Ease while walking; snap when far behind (fast forward, or a new episode)
        if (!this.camPos || this.camPos.distanceTo(want) > 3) this.camPos = want.clone();
        else this.camPos.lerp(want, 0.12);
        this.camera.position.copy(this.camPos);
        this.camera.lookAt(this.target.x + 0.4, this.target.y, 0.85);
        this.floorGroup.position.set(Math.round(this.target.x), Math.round(this.target.y), 0);
        this.sun.position.set(this.target.x + 3, this.target.y - 2, 6);
        this.sun.target.position.copy(this.target);
        this.renderer.render(this.scene, this.camera);
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        const dpr = this.renderer.getPixelRatio();
        this.mirror.getRenderTarget().setSize(Math.round(w * dpr * 0.5), Math.round(h * dpr * 0.5));
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }
}
