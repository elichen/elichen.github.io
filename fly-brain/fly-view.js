// Draws the tethered fly (flybody's meshes, posed by the worker's MuJoCo) and lets the visitor
// drag the drop. MuJoCo is z-up, in cm; the fly faces +x.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';

export const DROP_COLORS = { sugar: '#f4ead2', bitter: '#b9cf7a', water: '#a9d3f2', 'sugar+bitter': '#dcdca4' };
const WICK = new Set(['capillary', 'drop']);

export class FlyView {
    constructor(canvas, onWick) {
        this.canvas = canvas;
        this.onWick = onWick;
        const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        const scene = this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 0.6));
        const key = new THREE.DirectionalLight(0xffffff, 1.6);
        key.position.set(1, 0.6, 1.5);
        scene.add(key);

        this.camera = new THREE.PerspectiveCamera(30, 1, 0.005, 20);
        this.camera.up.set(0, 0, 1);
        // front right of the fly, a little below the proboscis; a light moves with the camera
        this.target = new THREE.Vector3(0.06, 0, -0.06);
        this.camera.position.set(0.32, -0.36, -0.12);
        const fill = new THREE.DirectionalLight(0xffffff, 1.4);
        fill.position.set(0.3, 0.5, 1);
        this.camera.add(fill);
        scene.add(this.camera);
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.target.copy(this.target);
        this.controls.enableDamping = true;
        this.controls.minDistance = 0.15;
        this.controls.maxDistance = 2.5;
        this.controls.update();
        this.parts = [];
        this.wickParts = [];
        this.dragging = false;
        this.wick = null;
        this.pointer();
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
    }

    setTaste(taste) {
        if (this.drop && taste) this.drop.mesh.material.color.set(DROP_COLORS[taste]).convertSRGBToLinear();
    }

    setBackground(color) {
        this.scene.background = new THREE.Color(color);
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

    // drag the drop in the plane through it that faces the camera; elsewhere, turn the view
    pointer() {
        const ray = new THREE.Raycaster();
        const plane = new THREE.Plane();
        const hit = new THREE.Vector3();
        const ndc = e => {
            const r = this.canvas.getBoundingClientRect();
            return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
        };
        let offset = null;
        this.canvas.addEventListener('pointerdown', e => {
            if (!this.wick) return;
            ray.setFromCamera(ndc(e), this.camera);
            if (!ray.intersectObjects(this.wickParts.map(p => p.mesh)).length) return;
            const at = new THREE.Vector3(...this.wick);
            plane.setFromNormalAndCoplanarPoint(this.camera.getWorldDirection(new THREE.Vector3()), at);
            ray.ray.intersectPlane(plane, hit);
            offset = at.sub(hit);
            this.dragging = true;
            this.controls.enabled = false;
            this.canvas.setPointerCapture(e.pointerId);
        });
        this.canvas.addEventListener('pointermove', e => {
            if (!this.dragging) return;
            ray.setFromCamera(ndc(e), this.camera);
            if (!ray.ray.intersectPlane(plane, hit)) return;
            const p = hit.add(offset);
            // keep it in front of the fly's head, within reach of the camera's view
            p.x = Math.min(0.6, Math.max(0.02, p.x));
            p.y = Math.min(0.3, Math.max(-0.3, p.y));
            p.z = Math.min(0.2, Math.max(-0.4, p.z));
            this.onWick([p.x, p.y, p.z]);
        });
        const up = () => { this.dragging = false; this.controls.enabled = true; };
        this.canvas.addEventListener('pointerup', up);
        this.canvas.addEventListener('pointercancel', up);
    }
}
