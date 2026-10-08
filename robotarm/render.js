// Draws the scene from MuJoCo's geom poses with three.js (MuJoCo is z-up): the
// Panda's visual meshes in their own materials, the box, and a floor with distance
// rings around the robot, the aim arrow, and where the throws landed.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { toCreasedNormals } from 'three/addons/utils/BufferGeometryUtils.js';
import { CENTER, BOX_RANGE } from './panda-env.js';

const RINGS = [0.5, 1, 1.5, 2, 2.5, 3];   // distance rings, meters from the base
const MARKS = 12;                         // landing marks kept on the floor

function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return new THREE.Color(v || fallback);
}

// A text label lying flat on the floor (or standing, as a sprite), `height` meters tall
function label(text, color, height, flat = true) {
    const c = document.createElement('canvas');
    const g = c.getContext('2d');
    const font = '600 64px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
    g.font = font;
    c.width = Math.ceil(g.measureText(text).width) + 16;
    c.height = 84;
    g.font = font;
    g.fillStyle = color.getStyle();
    g.textBaseline = 'middle';
    g.fillText(text, 8, 44);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    const w = height * c.width / c.height;
    if (!flat) {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
        s.scale.set(w, height, 1);
        return s;
    }
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, height), new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false }));
    return m;
}

// A flat arrow along +x from r0 to r1, `w` wide
function arrowShape(r0, r1, w) {
    const s = new THREE.Shape();
    const head = 0.08, hw = w * 3;
    s.moveTo(r0, -w / 2);
    s.lineTo(r1 - head, -w / 2);
    s.lineTo(r1 - head, -hw / 2);
    s.lineTo(r1, 0);
    s.lineTo(r1 - head, hw / 2);
    s.lineTo(r1 - head, w / 2);
    s.lineTo(r0, w / 2);
    s.closePath();
    return new THREE.ShapeGeometry(s);
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
        renderer.toneMappingExposure = 0.9;

        const scene = this.scene = new THREE.Scene();
        const pmrem = new THREE.PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
        scene.background = cssColor('--sky', '#eef0f2');
        scene.fog = new THREE.Fog(scene.background, 6, 18);

        this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 60);
        this.camera.up.set(0, 0, 1);
        this.camera.position.set(2.5, -2.5, 2.2);
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.target.set(0.1, 0, 0.15);
        this.controls.enableDamping = true;
        this.controls.minDistance = 0.8;
        this.controls.maxDistance = 7;
        this.controls.maxPolarAngle = Math.PI * 0.49;  // stay above the floor
        this.controls.update();

        scene.add(new THREE.HemisphereLight(0xffffff, 0x666666, 0.5));
        const sun = new THREE.DirectionalLight(0xffffff, 1.7);
        sun.position.set(1.2, -0.8, 2.6);
        sun.target.position.set(0.4, 0, 0);
        sun.castShadow = true;
        sun.shadow.mapSize.set(4096, 4096);
        Object.assign(sun.shadow.camera, { left: -3.2, right: 3.2, top: 3.2, bottom: -3.2, near: 0.5, far: 7 });
        sun.shadow.bias = -0.0004;
        sun.shadow.normalBias = 0.01;
        scene.add(sun, sun.target);

        // Floor, distance rings with labels, and the outline of the area new boxes drop into
        const ink = cssColor('--rings', '#9aa1a8');
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshStandardMaterial({ color: cssColor('--floor', '#806a41'), roughness: 1, envMapIntensity: 0.3 }));
        floor.receiveShadow = true;
        scene.add(floor);
        const ringMat = new THREE.MeshBasicMaterial({ color: ink, transparent: true, opacity: 0.28, depthWrite: false });
        for (const r of RINGS) {
            const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.003, r + 0.003, 192), ringMat);
            ring.position.z = 0.001;
            scene.add(ring);
            const t = label(`${r} m`, ink, 0.08);
            t.material.opacity = 0.6;
            t.position.set(r + 0.06, -0.06, 0.0015);
            scene.add(t);
        }
        const a = BOX_RANGE + 0.02;
        const pts = [[-a, -a], [a, -a], [a, a], [-a, a], [-a, -a]].map(([x, y]) => new THREE.Vector3(CENTER[0] + x, CENTER[1] + y, 0.0015));
        const area = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
            new THREE.LineDashedMaterial({ color: ink, dashSize: 0.02, gapSize: 0.015 }));
        area.computeLineDistances();
        scene.add(area);

        // The aim arrow, turned around the base each frame
        this.aim = new THREE.Mesh(arrowShape(0.3, 0.8, 0.022),
            new THREE.MeshBasicMaterial({ color: cssColor('--aim', '#46549b'), transparent: true, opacity: 0.85, depthWrite: false }));
        this.aim.position.z = 0.002;
        scene.add(this.aim);

        // Landing marks: recent ones fade; the best one gets a ring and a label
        this.markColor = cssColor('--mark', '#2f6db5');
        this.bestColor = cssColor('--best', '#3a9a5b');
        this.marks = [];
        this.markGeo = new THREE.CircleGeometry(0.018, 24);
        this.best = new THREE.Group();
        this.best.add(new THREE.Mesh(new THREE.RingGeometry(0.035, 0.055, 48), new THREE.MeshBasicMaterial({ color: this.bestColor, depthWrite: false, transparent: true })));
        this.best.visible = false;
        scene.add(this.best);
        this.lastLabel = null;

        // Robot: one mesh per visual geom (group 2), posed every frame
        const geometries = {};
        const materials = {};
        this.parts = [];
        for (let g = 0; g < m.ngeom; g++) {
            if (m.geom_group[g] !== 2 || m.geom_type[g] !== 7) continue;
            const id = m.geom_dataid[g];
            if (!geometries[id]) {
                const v0 = m.mesh_vertadr[id], nv = m.mesh_vertnum[id], f0 = m.mesh_faceadr[id], nf = m.mesh_facenum[id];
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(m.mesh_vert.slice(3 * v0, 3 * (v0 + nv))), 3));
                geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(m.mesh_face.slice(3 * f0, 3 * (f0 + nf))), 1));
                geometries[id] = toCreasedNormals(geo, Math.PI / 6);
            }
            const mat = m.geom_matid[g];
            if (!materials[mat]) {
                const c = m.mat_rgba.slice(4 * mat, 4 * mat + 3);
                materials[mat] = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace),
                    roughness: c[0] < 0.5 ? 0.5 : 0.35, metalness: 0,
                });
            }
            const mesh = new THREE.Mesh(geometries[id], materials[mat]);
            mesh.castShadow = mesh.receiveShadow = true;
            mesh.matrixAutoUpdate = false;
            scene.add(mesh);
            this.parts.push({ geom: g, mesh });
        }

        // The box (4 x 4 x 6 cm)
        const boxGeom = this.boxGeom = [...Array(m.ngeom).keys()].find(g => m.geom_bodyid[g] === env.boxBody);
        const s = m.geom_size.slice(3 * boxGeom, 3 * boxGeom + 3);
        this.box = new THREE.Mesh(new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]),
            new THREE.MeshStandardMaterial({ color: cssColor('--block', '#2f6db5'), roughness: 0.55 }));
        this.box.castShadow = this.box.receiveShadow = true;
        this.box.matrixAutoUpdate = false;
        scene.add(this.box);
        this.leaving = [];   // old boxes shrinking away when a new one drops

        this.resize();
        new ResizeObserver(() => this.resize()).observe(canvas);
    }

    resize() {
        const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
        if (!w || !h) return;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    // Pose a three.js object from MuJoCo's position (3) and rotation matrix (9, row-major)
    static pose(obj, p, r, i) {
        const e = obj.matrix.elements;  // column-major
        e[0] = r[9 * i]; e[4] = r[9 * i + 1]; e[8] = r[9 * i + 2];
        e[1] = r[9 * i + 3]; e[5] = r[9 * i + 4]; e[9] = r[9 * i + 5];
        e[2] = r[9 * i + 6]; e[6] = r[9 * i + 7]; e[10] = r[9 * i + 8];
        e[12] = p[3 * i]; e[13] = p[3 * i + 1]; e[14] = p[3 * i + 2];
        e[3] = e[7] = e[11] = 0; e[15] = 1;
        obj.matrixWorldNeedsUpdate = true;
    }

    // The old box shrinks away where it is (a new one has just been placed)
    boxLeaves() {
        const ghost = this.box.clone();
        ghost.material = this.box.material.clone();
        ghost.material.transparent = true;
        this.scene.add(ghost);
        this.leaving.push({ mesh: ghost, base: this.box.matrix.clone(), t: 0 });
    }

    // Mark where a throw landed, with its distance
    landed([x, y], distance, best) {
        const mark = new THREE.Mesh(this.markGeo, new THREE.MeshBasicMaterial({ color: this.markColor, transparent: true, depthWrite: false }));
        mark.position.set(x, y, 0.003);
        this.scene.add(mark);
        this.marks.push(mark);
        if (this.marks.length > MARKS) this.scene.remove(this.marks.shift());
        this.marks.forEach((m, i) => { m.material.opacity = 0.25 + 0.6 * (i + 1) / this.marks.length; });

        if (this.lastLabel) this.scene.remove(this.lastLabel.sprite);
        const sprite = label(`${distance.toFixed(2)} m`, best ? this.bestColor : this.markColor, 0.09, false);
        sprite.position.set(x, y, 0.16);
        this.scene.add(sprite);
        this.lastLabel = { sprite, t: 0 };
        if (best) {
            this.best.position.set(x, y, 0.004);
            this.best.visible = true;
        }
    }

    draw(dt, heading) {
        const d = this.env.data;
        const xpos = d.geom_xpos, xmat = d.geom_xmat;
        for (const { geom, mesh } of this.parts) SceneRenderer.pose(mesh, xpos, xmat, geom);
        SceneRenderer.pose(this.box, xpos, xmat, this.boxGeom);
        this.aim.rotation.z = heading;

        for (const l of this.leaving) {
            l.t += dt / 0.3;
            const k = Math.max(0, 1 - l.t);
            l.mesh.material.opacity = k;
            l.mesh.matrix.copy(l.base).multiply(new THREE.Matrix4().makeScale(k + 0.01, k + 0.01, k + 0.01));
            l.mesh.matrixWorldNeedsUpdate = true;
            if (l.t >= 1) this.scene.remove(l.mesh);
        }
        this.leaving = this.leaving.filter(l => l.t < 1);

        if (this.lastLabel) {
            const l = this.lastLabel;
            l.t += dt;
            l.sprite.material.opacity = Math.min(1, Math.max(0, (3 - l.t) / 0.6));
            if (l.t > 3) { this.scene.remove(l.sprite); this.lastLabel = null; }
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
