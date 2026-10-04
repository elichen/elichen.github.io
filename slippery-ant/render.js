// Draws the Ant from MuJoCo's geom poses with three.js. The floor follows the
// Ant (snapped to whole tiles so the grid looks fixed) and blends between
// rubber and ice with the friction.
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const MU_ICE = 0.02, MU_RUBBER = 2.0;

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
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.anisotropy = 8;
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
}

export class AntRenderer {
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

        // Floor
        const tex = floorTexture();
        tex.repeat.set(60, 60);
        this.floorMat = new THREE.MeshStandardMaterial({ map: tex });
        this.floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), this.floorMat);
        this.floor.receiveShadow = true;
        scene.add(this.floor);

        // Ant: one mesh per geom, posed from MuJoCo every frame
        this.antMat = new THREE.MeshStandardMaterial({ color: this.colors.ant, roughness: 0.45, metalness: 0.05 });
        this.meshes = [];
        for (let i = 0; i < info.ngeom; i++) {
            const type = info.geomType[i], [r, half] = info.geomSize.slice(3 * i, 3 * i + 2);
            let geo = null;
            if (type === 2) geo = new THREE.SphereGeometry(r, 32, 24);
            if (type === 3) geo = new THREE.CapsuleGeometry(r, 2 * half, 6, 16).rotateX(Math.PI / 2);  // MuJoCo capsules run along local z
            if (!geo) { this.meshes.push(null); continue; }
            const mesh = new THREE.Mesh(geo, this.antMat);
            mesh.matrixAutoUpdate = false;
            mesh.castShadow = true;
            scene.add(mesh);
            this.meshes.push(mesh);
            // Eyes on the torso, so you can tell which way it faces (+x is forward)
            if (type === 2) {
                const eyeMat = new THREE.MeshStandardMaterial({ color: this.colors.eye, roughness: 0.2 });
                for (const y of [-0.09, 0.09]) {
                    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), eyeMat);
                    eye.position.set(0.21, y, 0.1);
                    mesh.add(eye);
                }
            }
        }
        this.target = new THREE.Vector3();
        this.camPos = null;
        this.setFriction(MU_RUBBER);
        this.resize();
    }

    readColors() {
        this.colors = {
            bg: cssColor('--viewport-bg', '#dfe7ec'),
            ant: cssColor('--ant', '#3b2a22'),
            eye: cssColor('--ant-eye', '#f4f1ea'),
            ice: cssColor('--ice', '#d8ecf5'),
            rubber: cssColor('--rubber', '#3a3d40')
        };
        this.scene.background = this.colors.bg;
        this.scene.fog = new THREE.Fog(this.colors.bg, 12, 26);
    }

    // 1 = rubber, 0 = ice, on a log scale of friction
    setFriction(mu) {
        const t = Math.min(1, Math.max(0, Math.log(mu / MU_ICE) / Math.log(MU_RUBBER / MU_ICE)));
        this.floorMat.color.copy(this.colors.ice).lerp(this.colors.rubber, t);
        this.floorMat.roughness = 0.08 + 0.87 * t;
        this.floorMat.envMapIntensity = 1.2 - 0.9 * t;
        this.floorMat.needsUpdate = true;
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
        const [x, y] = frame.torso;
        this.target.set(x, y, 0.4);
    }

    render() {
        // Follow from behind and to the side, with a little lag
        const want = new THREE.Vector3(this.target.x - 2.1, this.target.y - 2.8, 1.75);
        // Ease while walking; snap when far behind (fast forward, or a new episode)
        if (!this.camPos || this.camPos.distanceTo(want) > 3) this.camPos = want.clone();
        else this.camPos.lerp(want, 0.12);
        this.camera.position.copy(this.camPos);
        this.camera.lookAt(this.target.x + 0.35, this.target.y, 0.2);
        this.floor.position.set(Math.round(this.target.x), Math.round(this.target.y), 0);
        this.sun.position.set(this.target.x + 3, this.target.y - 2, 6);
        this.sun.target.position.copy(this.target);
        this.renderer.render(this.scene, this.camera);
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }
}
