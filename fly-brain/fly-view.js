// Draws the walking fly (flybody's meshes, posed by the worker's MuJoCo) on a floor, follows it, and lets
// the visitor drag the drop. MuJoCo is z-up, in cm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

export const DROP_COLORS = { sugar: '#f4ead2', bitter: '#b9cf7a', water: '#a9d3f2', 'sugar+bitter': '#dcdca4' };
const WICK = new Set(['drop']);
const ODOR_RADIUS = 3;      // cm, the drawn smell; the rule's length scale is 1 cm (forage.js)

export class FlyView {
    constructor(canvas, onWick) {
        this.canvas = canvas;
        this.onWick = onWick;
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
        this.buildFloor();

        this.camera = new THREE.PerspectiveCamera(30, 1, 0.01, 80);
        this.camera.up.set(0, 0, 1);
        // behind and to the right of where the fly starts, looking down; a light moves with the camera
        this.target = new THREE.Vector3(0.3, 0, 0.05);
        this.camera.position.set(-0.9, -1.5, 1.0);
        const fill = new THREE.DirectionalLight(0xffffff, 1.2);
        fill.position.set(0.3, 0.5, 1);
        this.camera.add(fill);
        scene.add(this.camera);
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.target.copy(this.target);
        this.controls.enableDamping = true;
        this.controls.minDistance = 0.3;
        this.controls.maxDistance = 12;
        this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
        this.controls.update();
        this.follow = 0.08;     // share of the way to the fly per pose
        this.parts = [];
        this.wickParts = [];
        this.dragging = false;
        this.wick = null;
        this.pointer();
        new ResizeObserver(() => this.resize()).observe(canvas);
        this.resize();
    }

    // a floor with a 1 cm grid, and the drawn smell around the drop
    buildFloor() {
        const c = document.createElement('canvas');
        c.width = c.height = 256;
        const g = c.getContext('2d');
        g.strokeStyle = 'rgba(128,128,128,0.45)';
        g.lineWidth = 3;
        g.strokeRect(0, 0, 256, 256);
        const tex = new THREE.CanvasTexture(c);
        tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
        tex.repeat.set(200, 200);
        tex.anisotropy = 8;
        this.floorMat = new THREE.MeshStandardMaterial({ roughness: 0.95, envMapIntensity: 0.3 });
        const base = this.floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), this.floorMat);
        base.receiveShadow = true;
        const grid = this.grid = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
        grid.position.z = 0.0005;
        const o = document.createElement('canvas');
        o.width = o.height = 256;
        const og = o.getContext('2d');
        const rad = og.createRadialGradient(128, 128, 0, 128, 128, 128);
        rad.addColorStop(0, 'rgba(255,255,255,0.55)');
        rad.addColorStop(0.35, 'rgba(255,255,255,0.14)');
        rad.addColorStop(1, 'rgba(255,255,255,0)');
        og.fillStyle = rad;
        og.fillRect(0, 0, 256, 256);
        this.odor = new THREE.Mesh(new THREE.PlaneGeometry(2 * ODOR_RADIUS, 2 * ODOR_RADIUS),
            new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(o), transparent: true, depthWrite: false }));
        this.odor.position.z = 0.001;
        this.odor.visible = false;
        this.scene.add(base, grid, this.odor);
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
            if (g.name === 'drop') Object.assign(mat, { roughness: 0.08, envMapIntensity: 1.5 });
            const mesh = new THREE.Mesh(geo, mat);
            mesh.matrixAutoUpdate = false;
            mesh.frustumCulled = false;
            mesh.castShadow = g.name !== 'drop';
            this.scene.add(mesh);
            const part = { g: g.g, mesh, name: g.name };
            this.parts.push(part);
            if (WICK.has(g.name)) this.wickParts.push(part);
            if (g.name === 'drop') this.drop = part;
        }
    }

    // pose every geom from MuJoCo's geom_xpos / geom_xmat (row-major)
    update(m) {
        this.wick = m.wick;
        for (const { g, mesh } of this.parts) {
            const e = mesh.matrix.elements, r = m.xmat, p = m.xpos;
            e[0] = r[9 * g]; e[4] = r[9 * g + 1]; e[8] = r[9 * g + 2];
            e[1] = r[9 * g + 3]; e[5] = r[9 * g + 4]; e[9] = r[9 * g + 5];
            e[2] = r[9 * g + 6]; e[6] = r[9 * g + 7]; e[10] = r[9 * g + 8];
            e[12] = p[3 * g]; e[13] = p[3 * g + 1]; e[14] = p[3 * g + 2];
            e[3] = e[7] = e[11] = 0; e[15] = 1;
            mesh.matrixWorldNeedsUpdate = true;
        }
        for (const { mesh } of this.wickParts) mesh.visible = !!m.wick;
        this.odor.visible = !!m.wick;
        if (m.wick) this.odor.position.set(m.wick[0], m.wick[1], 0.001);
        // follow the fly, keeping the viewing angle
        if (m.fly) {
            const t = this.controls.target;
            const dx = (m.fly[0] - t.x) * this.follow, dy = (m.fly[1] - t.y) * this.follow;
            t.x += dx; t.y += dy;
            this.camera.position.x += dx; this.camera.position.y += dy;
            this.key.target.position.set(m.fly[0], m.fly[1], 0);
            this.key.position.set(m.fly[0] + 1, m.fly[1] + 0.6, 3);
        }
    }

    setTaste(taste) {
        if (!taste) return;
        const c = new THREE.Color(DROP_COLORS[taste]);
        if (this.drop) this.drop.mesh.material.color.copy(c);
        this.odor.material.color.copy(c);
    }

    setBackground(color) {
        const c = new THREE.Color(color);
        this.scene.background = c;
        this.scene.fog = new THREE.Fog(c, 8, 30);
        this.floorMat.color.copy(c);
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    render() {
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // drag the drop across the floor; elsewhere, turn the view
    pointer() {
        const ray = new THREE.Raycaster();
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        const hit = new THREE.Vector3();
        const ndc = e => {
            const r = this.canvas.getBoundingClientRect();
            return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        };
        this.canvas.addEventListener('pointerdown', e => {
            if (!this.wick) return;
            ray.setFromCamera(ndc(e), this.camera);
            if (!ray.ray.intersectPlane(plane, hit)) return;
            // the drop is small: anywhere within 1.5x its radius grabs it
            if (Math.hypot(hit.x - this.wick[0], hit.y - this.wick[1]) > 0.3) return;
            this.dragging = true;
            this.controls.enabled = false;
            this.canvas.setPointerCapture(e.pointerId);
        });
        this.canvas.addEventListener('pointermove', e => {
            if (!this.dragging) return;
            ray.setFromCamera(ndc(e), this.camera);
            if (ray.ray.intersectPlane(plane, hit)) this.onWick([hit.x, hit.y]);
        });
        const up = () => { this.dragging = false; this.controls.enabled = true; };
        this.canvas.addEventListener('pointerup', up);
        this.canvas.addEventListener('pointercancel', up);
    }
}
