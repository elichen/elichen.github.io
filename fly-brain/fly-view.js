// Draws the fly (flybody's meshes, posed by the worker's MuJoCo) and follows it. MuJoCo is z-up, in cm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

export class FlyView {
    constructor(canvas) {
        this.canvas = canvas;
        const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        const scene = this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6));
        renderer.shadowMap.enabled = true;
        const key = this.key = new THREE.DirectionalLight(0xffffff, 1.2);
        key.position.set(1, 0.6, 3);
        key.castShadow = true;
        key.shadow.mapSize.set(1024, 1024);
        Object.assign(key.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6, near: 0.1, far: 8 });
        key.shadow.bias = -0.0005;
        scene.add(key, key.target);

        this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 80);
        this.camera.up.set(0, 0, 1);
        // a light moves with the camera
        const fill = new THREE.DirectionalLight(0xffffff, 1.2);
        fill.position.set(0.3, 0.5, 1);
        this.camera.add(fill);
        scene.add(this.camera);
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.minDistance = 0.3;
        this.controls.maxDistance = 12;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
        this.followTau = 0.12;  // s: how fast the view catches up with the fly
        this.flyAt = null;
        this.drawnAt = performance.now();
        this.parts = [];
        this.shift = 0; this.shiftNow = 0;
        new ResizeObserver(() => this.resize()).observe(canvas);
        this.resize();
    }

    build({ geoms, meshes }) {
        const geometries = {};
        for (const k in meshes) {
            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(meshes[k].vert, 3));
            geo.setIndex(new THREE.BufferAttribute(meshes[k].face, 1));
            geometries[k] = toCreasedNormals(geo, Math.PI / 5);
        }
        for (const g of geoms) {
            let geo;
            if (g.mesh >= 0) geo = geometries[g.mesh];
            else if (g.type === 2) geo = new THREE.SphereGeometry(g.size[0], 32, 16);
            else if (g.type === 4) {
                geo = new THREE.SphereGeometry(1, 32, 16);
                geo.scale(g.size[0], g.size[1], g.size[2]);
            }
            else if (g.type === 3) {
                // capsule: MuJoCo's runs along its local z
                geo = new THREE.CapsuleGeometry(g.size[0], 2 * g.size[1], 8, 16);
                geo.rotateX(Math.PI / 2);
            } else continue;
            const [r, gr, b, a] = g.rgba;
            const mat = new THREE.MeshStandardMaterial({
                color: new THREE.Color().setRGB(r, gr, b, THREE.SRGBColorSpace),
                roughness: 0.45, metalness: 0, transparent: a < 1, opacity: a, depthWrite: a >= 1,
            });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.matrixAutoUpdate = false;
            mesh.frustumCulled = false;
            mesh.castShadow = true;
            this.scene.add(mesh);
            this.parts.push({ g: g.g, mesh });
        }
    }

    // pose every geom from MuJoCo's geom_xpos / geom_xmat (row-major)
    update(m) {
        for (const { g, mesh } of this.parts) {
            const e = mesh.matrix.elements, r = m.xmat, p = m.xpos;
            e[0] = r[9 * g]; e[4] = r[9 * g + 1]; e[8] = r[9 * g + 2];
            e[1] = r[9 * g + 3]; e[5] = r[9 * g + 4]; e[9] = r[9 * g + 5];
            e[2] = r[9 * g + 6]; e[6] = r[9 * g + 7]; e[10] = r[9 * g + 8];
            e[12] = p[3 * g]; e[13] = p[3 * g + 1]; e[14] = p[3 * g + 2];
            e[3] = e[7] = e[11] = 0; e[15] = 1;
            mesh.matrixWorldNeedsUpdate = true;
        }
        this.flyAt = m.fly;
        this.key.target.position.set(m.fly[0], m.fly[1], 0);
        this.key.position.set(m.fly[0] + 1, m.fly[1] + 0.6, 3);
    }

    // follow the fly, keeping the viewing angle. Per drawn frame, by time: it does not depend on how often poses arrive, and it still finishes while paused
    followFly() {
        const now = performance.now(), share = 1 - Math.exp(-Math.min(0.1, (now - this.drawnAt) / 1000) / this.followTau);
        this.drawnAt = now;
        if (!this.flyAt) return;
        const t = this.controls.target, dx = (this.flyAt[0] - t.x) * share, dy = (this.flyAt[1] - t.y) * share, dz = (this.flyAt[2] - t.z) * share;
        t.x += dx; t.y += dy; t.z += dz;
        this.camera.position.x += dx; this.camera.position.y += dy; this.camera.position.z += dz;
    }

    // the fly sits off centre while text covers part of the window
    applyShift() {
        this.shiftNow += (this.shift - this.shiftNow) * 0.08;
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (Math.abs(this.shiftNow) < 0.5 && !this.shift) { if (this.camera.view?.enabled) this.camera.clearViewOffset(); return; }
        this.camera.setViewOffset(w, h, this.shiftNow, 0, w, h);
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    render() {
        this.followFly();
        this.applyShift();
        this.controls.update();
        // the near plane scales with the zoom: a fixed 0.01 cm leaves too little depth precision far out, and the towel and coffee ring flicker through the table
        const near = Math.max(0.01, this.camera.position.distanceTo(this.controls.target) / 100);
        if (near !== this.camera.near) { this.camera.near = near; this.camera.updateProjectionMatrix(); }
        this.renderer.render(this.scene, this.camera);
    }

}
