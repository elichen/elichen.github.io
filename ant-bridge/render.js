// three.js view of the simulation: a forked twig in a rainforest, army ants with
// articulated legs, and a macro lens (shallow depth of field). Units are cm.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { buildAntGeometries, LEGS, ANTENNA, RIG_LENGTH, BODY_HEIGHT } from './ant-model.js';

const MAX_ANTS = 700;

// Ant colours sampled from Brian Gratwicke's photo of Eciton workers carrying brood
// (Wikimedia Commons, CC BY 2.0): lit cuticle #d88551, mid-tone #9f593a, legs in
// shade #613823, brood #eee2d8. Albedo sits between the lit and mid tones.
const ANT_BODY = '#a8461a';
const ANT_LEG = '#8c3e1b';
const BROOD = '#eee2d8';

const _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _p = new THREE.Vector3();
const _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3();
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3(), _d = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _basis = new THREE.Matrix4();

export class View {
    constructor(canvas) {
        this.canvas = canvas;
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowMap;
        this.renderer.toneMapping = THREE.NoToneMapping;   // tone mapping happens after depth of field
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(28, 1, 0.3, 600);
        this.controls = new OrbitControls(this.camera, canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 3;
        this.controls.maxDistance = 70;
        // stay above the twig, looking down at the forest floor; the horizon is never in view
        this.controls.maxPolarAngle = Math.PI * 0.38;
        this.controls.addEventListener('start', () => { this.userAt = performance.now(); this.userHolding = true; });
        this.controls.addEventListener('end', () => { this.userAt = performance.now(); this.userHolding = false; });
        this.userAt = -1e9;
        this.antState = new Map();
        this.time = 0;
        this.aperture = 1;
        this.ready = this.load();
    }

    async load() {
        const tex = new THREE.TextureLoader();
        const load = (f, srgb = false) => new Promise((res, rej) => tex.load(f, (t) => {
            t.wrapS = t.wrapT = THREE.RepeatWrapping;
            t.anisotropy = 8;
            if (srgb) t.colorSpace = THREE.SRGBColorSpace;
            res(t);
        }, undefined, rej));
        const hdr = new Promise((res, rej) => new HDRLoader().load('env/rainforest_trail_1k.hdr', res, undefined, rej));
        const [env, barkD, barkN, barkR, litD, litN, mossD] = await Promise.all([
            hdr, load('env/bark_diff.jpg', true), load('env/bark_nor.jpg'), load('env/bark_rough.jpg'),
            load('env/litter_diff.jpg', true), load('env/litter_nor.jpg'), load('env/moss_diff.jpg', true),
        ]);
        env.mapping = THREE.EquirectangularReflectionMapping;
        this.scene.environment = env;
        this.scene.environmentIntensity = 0.45;
        this.scene.background = env;
        this.scene.backgroundBlurriness = 0.35;
        this.scene.backgroundIntensity = 0.3;
        this.scene.environmentRotation.y = this.scene.backgroundRotation.y = 1.2;
        this.tex = { barkD, barkN, barkR, litD, litN, mossD };
        this.buildLights();
        this.buildFloor();
        this.buildLeaves();
        this.buildAnts();
        this.buildPost();
    }

    // --- world

    buildLights() {
        // the sun through a gap in the canopy; its shadow box follows the bridge
        const sun = new THREE.DirectionalLight('#fff0d8', 3.2);
        sun.position.set(-40, 70, -25);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        const sc = sun.shadow.camera;
        sc.left = -9; sc.right = 9; sc.top = 9; sc.bottom = -9; sc.near = 40; sc.far = 140;
        sun.shadow.bias = -0.0002;
        sun.shadow.normalBias = 0.01;
        this.scene.add(sun, sun.target);
        this.sun = sun;
        this.sunOffset = sun.position.clone();
        this.cookie = this.canopyCookie();
        const sky = new THREE.HemisphereLight('#d6e6cf', '#3b2a1a', 0.25);
        this.scene.add(sky);
    }

    canopyCookie() {
        const c = document.createElement('canvas');
        c.width = c.height = 512;
        const g = c.getContext('2d');
        g.fillStyle = '#000';
        g.fillRect(0, 0, 512, 512);
        let seed = 7;
        const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        g.filter = 'blur(6px)';
        // a bright gap in the middle (so the twig is in sun) and scattered smaller gaps
        for (let i = 0; i < 120; i++) {
            const centre = i < 14;
            const r = centre ? 40 + rnd() * 60 : 6 + rnd() * 26;
            const x = centre ? 256 + (rnd() - 0.5) * 120 : rnd() * 512;
            const y = centre ? 256 + (rnd() - 0.5) * 120 : rnd() * 512;
            g.fillStyle = `rgba(255,255,255,${centre ? 0.9 : 0.35 + rnd() * 0.5})`;
            g.beginPath();
            g.ellipse(x, y, r, r * (0.6 + rnd() * 0.6), rnd() * 3, 0, Math.PI * 2);
            g.fill();
        }
        g.filter = 'blur(3px)';
        // leaves inside the gap
        for (let i = 0; i < 40; i++) {
            const x = 256 + (rnd() - 0.5) * 300, y = 256 + (rnd() - 0.5) * 300;
            g.fillStyle = `rgba(0,0,0,${0.5 + rnd() * 0.4})`;
            g.beginPath();
            g.ellipse(x, y, 6 + rnd() * 16, 3 + rnd() * 6, rnd() * 3, 0, Math.PI * 2);
            g.fill();
        }
        const t = new THREE.CanvasTexture(c);
        t.colorSpace = THREE.SRGBColorSpace;
        return t;
    }

    buildFloor() {
        const { litD, litN } = this.tex;
        litD.repeat.set(40, 40); litN.repeat.set(40, 40);
        const fm = new THREE.MeshStandardMaterial({ map: litD, normalMap: litN, roughness: 0.92, color: '#8f8270' });
        fm.onBeforeCompile = (sh) => {
            sh.uniforms.cookie = { value: this.cookie };
            sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec2 vFloor;')
                .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvFloor = (modelMatrix * vec4(transformed, 1.0)).xz;');
            sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform sampler2D cookie;\nvarying vec2 vFloor;')
                .replace('#include <map_fragment>', '#include <map_fragment>\ndiffuseColor.rgb *= 0.35 + 1.6 * texture2D(cookie, vFloor / 90.0 + 0.5).r;');
        };
        const floor = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), fm);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -40;
        floor.receiveShadow = true;
        this.scene.add(floor);
    }

    // Broad rainforest leaves below and behind the twig. They are far out of focus and
    // turn into soft green shapes and bright specular bokeh.
    buildLeaves() {
        const c = document.createElement('canvas');
        c.width = 512; c.height = 512;
        const g = c.getContext('2d');
        const greens = ['#24401a', '#2f4f1e', '#3a5a22', '#1f3516', '#46622a', '#4a4222'];
        let seed = 11;
        const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
        // four leaf shapes in a 2x2 atlas
        for (let k = 0; k < 4; k++) {
            const ox = (k % 2) * 256 + 128, oy = Math.floor(k / 2) * 256 + 128;
            g.save(); g.translate(ox, oy); g.rotate(-Math.PI / 2);
            const L = 118, W = 34 + rnd() * 30;
            g.beginPath(); g.moveTo(-L, 0);
            g.bezierCurveTo(-L * 0.4, -W * 1.2, L * 0.5, -W, L, 0);
            g.bezierCurveTo(L * 0.5, W, -L * 0.4, W * 1.2, -L, 0);
            g.fillStyle = greens[k % greens.length]; g.fill();
            g.strokeStyle = 'rgba(200,230,150,0.35)'; g.lineWidth = 2.5;
            g.beginPath(); g.moveTo(-L, 0); g.lineTo(L * 0.95, 0); g.stroke();
            g.lineWidth = 1.2;
            for (let v = -0.8; v < 0.9; v += 0.16) {
                g.beginPath(); g.moveTo(L * v, 0); g.lineTo(L * (v + 0.18), -W * 0.8); g.moveTo(L * v, 0); g.lineTo(L * (v + 0.18), W * 0.8); g.stroke();
            }
            g.restore();
        }
        const tex = new THREE.CanvasTexture(c);
        tex.colorSpace = THREE.SRGBColorSpace;
        const mat = new THREE.MeshStandardMaterial({ map: tex, alphaTest: 0.5, side: THREE.DoubleSide, roughness: 0.3, metalness: 0 });
        const geo = new THREE.PlaneGeometry(1, 1);
        const group = new THREE.Group();
        for (let i = 0; i < 90; i++) {
            const k = i % 4, u0 = (k % 2) * 0.5, v0 = 0.5 - Math.floor(k / 2) * 0.5;
            const gg = geo.clone();
            const uv = gg.attributes.uv;
            for (let j = 0; j < uv.count; j++) uv.setXY(j, u0 + uv.getX(j) * 0.5, v0 + uv.getY(j) * 0.5);
            const m = new THREE.Mesh(gg, mat);
            const r = 15 + rnd() * 70, a = rnd() * Math.PI * 2;
            const size = 6 + rnd() * 16;
            m.scale.set(size * 0.6, size, 1);
            m.position.set(Math.cos(a) * r, -40 + rnd() * (i < 20 ? 22 : 6), 12 + Math.sin(a) * r);
            m.rotation.set(-Math.PI / 2 + (rnd() - 0.5) * 1.2, (rnd() - 0.5) * 0.6, rnd() * Math.PI * 2);
            group.add(m);
        }
        this.scene.add(group);
    }

    // The forked twig. The simulation's tines are straight cylinders of radius r lying in
    // y = 0; past the simulated ends they curve down to the leaf litter.
    setFork(sim) {
        if (this.fork) { this.scene.remove(this.fork); this.fork.traverse(o => o.geometry?.dispose()); }
        const { barkD, barkN, barkR, mossD } = this.tex;
        const group = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({ map: barkD, normalMap: barkN, roughnessMap: barkR, roughness: 1, color: '#d8c4ab' });
        mat.normalScale.set(1.2, 1.2);
        // moss on the upper side in patches
        mat.onBeforeCompile = (sh) => {
            sh.uniforms.mossMap = { value: mossD };
            sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vWorldPos;\nvarying vec3 vWorldNrm;')
                .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\nvWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;\nvWorldNrm = normalize(mat3(modelMatrix) * objectNormal);');
            sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform sampler2D mossMap;\nvarying vec3 vWorldPos;\nvarying vec3 vWorldNrm;\n' +
                'float hash3(vec3 p){ return fract(sin(dot(p, vec3(12.9898,78.233,37.719))) * 43758.5453); }\n' +
                'float vnoise(vec3 p){ vec3 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);\n' +
                ' return mix(mix(mix(hash3(i),hash3(i+vec3(1,0,0)),f.x),mix(hash3(i+vec3(0,1,0)),hash3(i+vec3(1,1,0)),f.x),f.y),\n' +
                '  mix(mix(hash3(i+vec3(0,0,1)),hash3(i+vec3(1,0,1)),f.x),mix(hash3(i+vec3(0,1,1)),hash3(i+vec3(1,1,1)),f.x),f.y),f.z); }')
                .replace('#include <map_fragment>', '#include <map_fragment>\n' +
                    'float mossAmt = smoothstep(0.68, 0.82, vnoise(vWorldPos * 1.3) * 0.6 + vnoise(vWorldPos * 6.0) * 0.4) * smoothstep(0.1, 0.8, vWorldNrm.y);\n' +
                    'vec3 moss = texture2D(mossMap, vWorldPos.xz * 0.35).rgb;\n' +
                    'diffuseColor.rgb = mix(diffuseColor.rgb, moss * vec3(0.55, 0.62, 0.45), mossAmt * 0.6);');
        };
        const r = sim.p.twigRadius, L = sim.p.armLength;
        const tineCurve = (t) => {
            const pts = [];
            for (let s = -0.4; s <= L + 0.6; s += 0.5) pts.push(new THREE.Vector3(t.ax + t.dx * s, 0, t.az + t.dz * s));
            // past the end the twig bends down into the litter
            const ex = t.ax + t.dx * (L + 0.6), ez = t.az + t.dz * (L + 0.6);
            for (const [k, y] of [[2, -0.6], [4, -2.5], [6, -7], [8, -16], [9, -30], [9.5, -42]]) {
                pts.push(new THREE.Vector3(ex + t.dx * k, y, ez + t.dz * k));
            }
            return new THREE.CatmullRomCurve3(pts, false, 'catmullrom', 0.2);
        };
        const addTube = (curve, rad, len) => {
            const g = new THREE.TubeGeometry(curve, Math.ceil(len * 6), rad, 28, false);
            const uv = g.attributes.uv;
            for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * len / (2 * Math.PI * rad), uv.getY(i));
            const m = new THREE.Mesh(g, mat);
            m.castShadow = m.receiveShadow = true;
            group.add(m);
        };
        for (const k of [0, 1]) { const c = tineCurve(sim.twigs[k]); addTube(c, r, c.getLength()); }
        // the stem continues past the crotch, thickening, and dips away
        const st = sim.twigs[2];
        const stem = new THREE.CatmullRomCurve3([
            new THREE.Vector3(0, 0, st.az - 0.6), new THREE.Vector3(0, 0, st.az + 3), new THREE.Vector3(0.2, -0.3, st.az + 8),
            new THREE.Vector3(0.8, -1.5, st.az + 18), new THREE.Vector3(2, -5, st.az + 32), new THREE.Vector3(4, -14, st.az + 50)]);
        const sg = new THREE.TubeGeometry(stem, 120, 1, 32, false);
        const pos = sg.attributes.position, uv = sg.attributes.uv;
        // taper: radius grows along the stem
        const len = stem.getLength();
        for (let i = 0; i < pos.count; i++) {
            const u = uv.getX(i), c = stem.getPointAt(Math.min(1, u));
            const rr = st.r * (1 + u * 1.6);
            _a.set(pos.getX(i), pos.getY(i), pos.getZ(i)).sub(c).normalize().multiplyScalar(rr).add(c);
            pos.setXYZ(i, _a.x, _a.y, _a.z);
            uv.setXY(i, u * len / (2 * Math.PI * st.r * 1.5), uv.getY(i));
        }
        sg.computeVertexNormals();
        const sm = new THREE.Mesh(sg, mat);
        sm.castShadow = sm.receiveShadow = true;
        group.add(sm);
        // a swelling where the tines join
        const knot = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), mat);
        knot.scale.set(r * 1.25, r * 1.05, r * 1.6);
        knot.position.set(0, -0.02, sim.crotch.z + 0.15);
        knot.castShadow = knot.receiveShadow = true;
        group.add(knot);
        this.scene.add(group);
        this.fork = group;
        this.sim = sim;
        this.antState.clear();
        const target = new THREE.Vector3(0, 0.3, sim.crotch.z - 2.2);
        this.controls.target.copy(target);
        this.camera.position.set(target.x + 3.5, target.y + 6, target.z - 8);
        this.focus = target.clone();
        this.sun.target.position.copy(target);
        this.sun.position.copy(target).add(this.sunOffset);
    }

    buildAnts() {
        const geo = buildAntGeometries();
        const body = new THREE.MeshPhysicalMaterial({
            color: ANT_BODY, vertexColors: true, roughness: 0.32, clearcoat: 0.9, clearcoatRoughness: 0.12,
            sheen: 0.25, sheenColor: '#ff9a5a', sheenRoughness: 0.5,
        });
        const leg = new THREE.MeshPhysicalMaterial({
            color: ANT_LEG, vertexColors: true, roughness: 0.4, clearcoat: 0.6, clearcoatRoughness: 0.2,
            sheen: 0.2, sheenColor: '#ff9a5a',
        });
        const brood = new THREE.MeshPhysicalMaterial({ color: BROOD, vertexColors: true, roughness: 0.55, sheen: 0.6, sheenColor: '#ffffff', clearcoat: 0.2 });
        const mk = (g, m, count) => {
            const im = new THREE.InstancedMesh(g, m, count);
            im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            im.castShadow = true; im.receiveShadow = true;
            im.frustumCulled = false;
            im.count = 0;
            this.scene.add(im);
            return im;
        };
        this.parts = {
            head: mk(geo.head, body, MAX_ANTS), headMajor: mk(geo.headMajor, body, 40),
            mesosoma: mk(geo.mesosoma, body, MAX_ANTS), petiole: mk(geo.petiole, body, MAX_ANTS),
            postpetiole: mk(geo.postpetiole, body, MAX_ANTS), gaster: mk(geo.gaster, body, MAX_ANTS),
            coxa: mk(geo.coxa, leg, MAX_ANTS * 6), femur: mk(geo.femur, leg, MAX_ANTS * 6),
            tibia: mk(geo.tibia, leg, MAX_ANTS * 6), tarsus: mk(geo.tarsus, leg, MAX_ANTS * 6),
            scape: mk(geo.scape, leg, MAX_ANTS * 2), funiculus: mk(geo.funiculus, leg, MAX_ANTS * 2),
            brood: mk(geo.brood, brood, MAX_ANTS),
        };
    }

    // --- post: depth of field, then tone mapping, vignette and a little grain

    buildPost() {
        const quad = (frag, uniforms) => {
            const m = new THREE.ShaderMaterial({
                uniforms, fragmentShader: frag, depthTest: false, depthWrite: false,
                vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
            });
            const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), m);
            mesh.frustumCulled = false;
            const s = new THREE.Scene(); s.add(mesh);
            return { scene: s, mat: m };
        };
        this.postCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        this.dof = quad(DOF_FRAG, {
            tColor: { value: null }, tDepth: { value: null }, near: { value: 0.3 }, far: { value: 600 },
            focus: { value: 15 }, focusScale: { value: 10 }, pixel: { value: new THREE.Vector2() }, maxBlur: { value: 14 },
        });
        this.final = quad(FINAL_FRAG, {
            tColor: { value: null }, tBlur: { value: null }, tDepth: { value: null }, near: { value: 0.3 }, far: { value: 600 },
            focus: { value: 15 }, focusScale: { value: 10 }, maxBlur: { value: 14 },
            exposure: { value: 1.3 }, time: { value: 0 }, resolution: { value: new THREE.Vector2() },
        });
    }

    resize(w, h) {
        if (!this.dof) return;
        const pr = this.renderer.getPixelRatio();
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        const W = Math.floor(w * pr), H = Math.floor(h * pr);
        this.rtScene?.dispose(); this.rtDof?.dispose();
        this.rtScene = new THREE.WebGLRenderTarget(W, H, { type: THREE.HalfFloatType, samples: 4 });
        this.rtScene.depthTexture = new THREE.DepthTexture(W, H, THREE.FloatType);
        // the blur runs at half resolution; sharp regions come from the full-resolution image
        const hw = Math.ceil(W / 2), hh = Math.ceil(H / 2);
        this.rtDof = new THREE.WebGLRenderTarget(hw, hh, { type: THREE.HalfFloatType });
        this.dof.mat.uniforms.pixel.value.set(1 / hw, 1 / hh);
        // blur radius in half-resolution pixels, about 2.5% of the image width at most
        this.blurPx = Math.max(6, Math.min(24, Math.round(hw * 0.025)));
        this.dof.mat.uniforms.maxBlur.value = this.blurPx;
        this.final.mat.uniforms.maxBlur.value = this.blurPx * 2;
        this.final.mat.uniforms.resolution.value.set(W, H);
    }

    // --- ants

    instance(part, i, m) { part.setMatrixAt(i, m); }

    // matrix that carries +y (unit length) from a to b, with thickness k
    segMatrix(a, b, k) {
        _d.subVectors(b, a);
        const len = _d.length() || 1e-6;
        _q.setFromUnitVectors(_up, _d.multiplyScalar(1 / len));
        _s.set(k, len, k);
        return _m.compose(a, _q, _s);
    }

    // two-bone leg: hip → (coxa) → knee → ankle → foot, knee bent up and outward
    leg(ix, scale, hipW, footW, out, upW, L, flex) {
        const P = this.parts;
        // coxa points outward and down from the hip
        _a.copy(out).multiplyScalar(0.8).addScaledVector(upW, -0.6).normalize().multiplyScalar(L.coxa * scale).add(hipW);
        this.instance(P.coxa, ix, this.segMatrix(hipW, _a, scale));
        // tarsus lies from the ankle down to the foot, sloping outward
        const tl = L.tarsus * scale;
        _b.copy(footW).addScaledVector(out, -tl * 0.75).addScaledVector(upW, tl * (0.55 + flex));
        // femur + tibia from coxa end (_a) to ankle (_b)
        const f = L.femur * scale, t = L.tibia * scale;
        _c.subVectors(_b, _a);
        let dist = _c.length();
        const dmax = (f + t) * 0.995, dmin = Math.abs(f - t) + 1e-4;
        if (dist > dmax) { _c.multiplyScalar(dmax / dist); _b.copy(_a).add(_c); dist = dmax; }
        dist = Math.max(dist, dmin);
        _c.normalize();
        const along = (f * f - t * t + dist * dist) / (2 * dist);
        const hgt = Math.sqrt(Math.max(0, f * f - along * along));
        _d.copy(upW).multiplyScalar(0.45).add(out);
        _d.addScaledVector(_c, -_d.dot(_c)).normalize();
        const knee = _p.copy(_a).addScaledVector(_c, along).addScaledVector(_d, hgt);
        const kneeV = new THREE.Vector3().copy(knee);
        this.instance(P.femur, ix, this.segMatrix(_a, kneeV, scale));
        this.instance(P.tibia, ix, this.segMatrix(kneeV, _b, scale));
        this.instance(P.tarsus, ix, this.segMatrix(_b, footW, scale));
    }

    update(dt, sim = this.sim) {
        if (!this.parts || !sim) return;
        this.time += dt;
        const P = this.parts;
        let nA = 0, nMaj = 0, nLeg = 0, nAnt = 0, nBrood = 0;
        const seen = new Set();
        const state = (id) => {
            let s = this.antState.get(id);
            if (!s) { s = { n: new THREE.Vector3(0, 1, 0) }; this.antState.set(id, s); }
            seen.add(id);
            return s;
        };
        const footW = new THREE.Vector3(), hipW = new THREE.Vector3(), out = new THREE.Vector3();
        const t = this.time;

        const drawAnt = (id, scale, origin, X, Y, Z, caste, opts) => {
            if (nA >= MAX_ANTS) return;
            _basis.makeBasis(X, Y, Z);
            const body = new THREE.Matrix4().copy(_basis).setPosition(origin);
            const bodyS = new THREE.Matrix4().copy(body).scale(_s.set(scale, scale, scale));
            if (caste === 3 && nMaj < 40) {
                this.instance(P.headMajor, nMaj++, bodyS);
                this.instance(P.head, nA, _m.makeScale(0, 0, 0));
            } else this.instance(P.head, nA, bodyS);
            this.instance(P.mesosoma, nA, bodyS);
            this.instance(P.petiole, nA, bodyS);
            this.instance(P.postpetiole, nA, bodyS);
            // gaster bobs a little with the gait
            const g = new THREE.Matrix4().makeRotationZ(opts.gasterTilt).premultiply(new THREE.Matrix4().makeTranslation(-0.205, 0, 0));
            g.multiply(new THREE.Matrix4().makeTranslation(0.205, 0, 0));
            this.instance(P.gaster, nA, new THREE.Matrix4().multiplyMatrices(bodyS, g));
            if (opts.carry && nBrood < MAX_ANTS) {
                const bm = new THREE.Matrix4().makeRotationZ(-0.12).setPosition(0.1, -0.075, 0);
                this.instance(P.brood, nBrood++, new THREE.Matrix4().multiplyMatrices(bodyS, bm));
            }
            // legs
            for (let k = 0; k < 6; k++) {
                const L = LEGS[k];
                hipW.set(L.hip[0], L.hip[1], L.hip[2]).applyMatrix4(bodyS);
                opts.foot(k, L, footW);
                out.copy(footW).sub(hipW);
                out.addScaledVector(Y, -out.dot(Y));
                if (out.lengthSq() < 1e-10) out.copy(Z).multiplyScalar(L.side);
                out.normalize();
                this.leg(nLeg++, scale, hipW, footW, out, Y, L, opts.flex ?? 0);
            }
            // antennae: elbowed, the tips tapping ahead
            for (const s of [-1, 1]) {
                const base = _a.set(ANTENNA.base[0], ANTENNA.base[1], ANTENNA.base[2] * s).applyMatrix4(bodyS).clone();
                const sw = opts.antenna(s);
                const dirS = _b.copy(X).multiplyScalar(0.8).addScaledVector(Y, 0.3 + sw.lift).addScaledVector(Z, s * (0.55 + sw.spread)).normalize();
                const elbow = base.clone().addScaledVector(dirS, ANTENNA.scape * scale);
                this.instance(P.scape, nAnt, this.segMatrix(base, elbow, scale));
                const dirF = _c.copy(X).multiplyScalar(0.85).addScaledVector(Y, -0.55 + sw.tap).addScaledVector(Z, s * (0.12 + sw.spread * 0.5)).normalize();
                const tip = elbow.clone().addScaledVector(dirF, ANTENNA.funiculus * scale);
                this.instance(P.funiculus, nAnt++, this.segMatrix(elbow, tip, scale));
            }
            nA++;
        };

        // walking ants
        for (const w of sim.walkers) {
            const s = state(w.id);
            const scale = w.size / RIG_LENGTH;
            // surface normal from the slope of the ground under the ant
            const e = 0.12;
            const g0 = sim.groundAt(w.x, w.z);
            if (g0 > -Infinity) {
                const gx1 = sim.groundAt(w.x + e, w.z), gx0 = sim.groundAt(w.x - e, w.z);
                const gz1 = sim.groundAt(w.x, w.z + e), gz0 = sim.groundAt(w.x, w.z - e);
                if (gx1 > -Infinity && gx0 > -Infinity && gz1 > -Infinity && gz0 > -Infinity) {
                    _a.set(-(gx1 - gx0) / (2 * e), 1, -(gz1 - gz0) / (2 * e)).normalize();
                    s.n.lerp(_a, Math.min(1, dt * 12)).normalize();
                }
            }
            const Y = _y.copy(s.n);
            const X = _x.set(Math.cos(w.hd), 0, Math.sin(w.hd));
            X.addScaledVector(Y, -X.dot(Y)).normalize();
            const Z = _z.crossVectors(X, Y).normalize();
            const origin = _p.set(w.x, w.y - 0.13 * w.size + BODY_HEIGHT * scale, w.z);
            const holding = w.state === 1;
            const stride = holding ? 0 : 0.27 * Math.min(1, w.v / 3);
            const ph = w.phase;
            const Xc = X.clone(), Yc = Y.clone(), Zc = Z.clone(), Oc = origin.clone();
            const bob = Math.sin(ph * Math.PI * 4) * 0.008 * scale * Math.min(1, w.v / 3);
            Oc.addScaledVector(Yc, bob);
            drawAnt(w.id, scale, Oc, Xc, Yc, Zc, w.caste, {
                carry: w.carry,
                gasterTilt: holding ? -0.15 : Math.sin(ph * Math.PI * 4) * 0.04,
                flex: holding ? 0.3 : 0,
                foot: (k, L, f) => {
                    const gp = (ph + (L.group ? 0.5 : 0)) % 1;
                    let xo, lift;
                    if (gp < 0.5) { const u = gp / 0.5; xo = stride / 2 - u * stride; lift = 0; }
                    else { const u = (gp - 0.5) / 0.5; xo = -stride / 2 + u * stride; lift = Math.sin(Math.PI * u) * 0.07 * Math.min(1, stride * 6); }
                    const spread = holding ? 1.25 : 1;
                    f.set(L.rest[0] * spread + xo, -BODY_HEIGHT + lift, L.rest[1] * spread);
                    f.applyMatrix4(_m.makeBasis(Xc, Yc, Zc).setPosition(Oc).scale(_s.set(scale, scale, scale)));
                    const gy = sim.groundAt(f.x, f.z);
                    if (gy > -Infinity && Yc.y > 0.6) f.y = gy + lift * scale + 0.01;
                },
                antenna: (sd) => ({
                    lift: 0.1 * Math.sin(t * 9 + w.id + sd),
                    tap: 0.25 * Math.sin(t * 13 + w.id * 1.7 + sd * 2),
                    spread: 0.15 * Math.sin(t * 7 + w.id * 0.9 - sd),
                }),
            });
        }

        // bridge ants: bodies from the simulation's two particles, legs reaching for what they hold
        const PX = sim.px;
        for (const a of sim.bridge) {
            const s = state(a.id);
            const scale = a.size / RIG_LENGTH;
            const F = new THREE.Vector3(PX[a.f * 3], PX[a.f * 3 + 1], PX[a.f * 3 + 2]);
            const B = new THREE.Vector3(PX[a.b * 3], PX[a.b * 3 + 1], PX[a.b * 3 + 2]);
            const X = new THREE.Vector3().subVectors(F, B).normalize();
            const Y = new THREE.Vector3(0, 1, 0).addScaledVector(X, -X.y).normalize().applyAxisAngle(X, a.roll);
            const Z = new THREE.Vector3().crossVectors(X, Y).normalize();
            const origin = new THREE.Vector3().addVectors(F, B).multiplyScalar(0.5).addScaledVector(X, 0.065 * scale).addScaledVector(Y, 0.02 * scale);
            // what this ant can hold: bark it grips and the bodies of ants it is linked to
            const holds = [];
            for (const gr of a.grips) {
                if (gr.anchor) holds.push(new THREE.Vector3(...gr.anchor));
                else holds.push(new THREE.Vector3(PX[gr.to * 3], PX[gr.to * 3 + 1], PX[gr.to * 3 + 2]));
            }
            const bodyS = new THREE.Matrix4().makeBasis(X, Y, Z).setPosition(origin).scale(_s.set(scale, scale, scale));
            const used = new Map();
            drawAnt(a.id, scale, origin, X, Y, Z, a.caste, {
                carry: false,
                gasterTilt: 0.12 * Math.sin(t * 0.7 + a.seed * 20),
                flex: 0.15,
                foot: (k, L, f) => {
                    // reach out from the body toward the nearest hold on this side
                    const want = new THREE.Vector3(L.rest[0] * 1.25, -BODY_HEIGHT * 0.6, L.rest[1] * 1.35).applyMatrix4(bodyS);
                    let best = null, bd = (L.femur + L.tibia + L.tarsus) * scale * 1.15;
                    for (let h = 0; h < holds.length; h++) {
                        const d = holds[h].distanceTo(want) + (used.get(h) || 0) * 0.08 * scale;
                        if (d < bd) { bd = d; best = h; }
                    }
                    if (best !== null) {
                        used.set(best, (used.get(best) || 0) + 1);
                        const jit = 0.06 * scale;
                        f.copy(holds[best]).add(_a.set(Math.sin(a.seed * 31 + k * 7) * jit, 0.03 * scale, Math.cos(a.seed * 17 + k * 5) * jit));
                        f.lerp(want, 0.25);
                    } else {
                        f.copy(want).addScaledVector(_up, -0.05 * scale);
                    }
                    // slow shifting of the grip
                    f.y += Math.sin(t * 1.3 + a.seed * 40 + k) * 0.01 * scale;
                },
                antenna: (sd) => ({
                    lift: -0.25 + 0.06 * Math.sin(t * 1.1 + a.seed * 9 + sd),
                    tap: -0.2 + 0.12 * Math.sin(t * 2.3 + a.seed * 13 + sd * 2),
                    spread: 0.2,
                }),
            });
        }
        for (const id of this.antState.keys()) if (!seen.has(id)) this.antState.delete(id);

        const setCount = (im, n) => { im.count = n; im.instanceMatrix.needsUpdate = true; };
        for (const k of ['head', 'mesosoma', 'petiole', 'postpetiole', 'gaster']) setCount(P[k], nA);
        setCount(P.headMajor, nMaj);
        for (const k of ['coxa', 'femur', 'tibia', 'tarsus']) setCount(P[k], nLeg);
        setCount(P.scape, nAnt); setCount(P.funiculus, nAnt);
        setCount(P.brood, nBrood);
    }

    // --- frame

    // keep the camera's target and the focus on the bridge unless the user is steering
    follow(dt, sim) {
        const m = sim.bridge.length ? sim.bridge : null;
        const goal = new THREE.Vector3(0, 0.3, sim.crotch.z - 2.2);
        if (m) {
            let x = 0, y = 0, z = 0;
            for (const a of m) { x += sim.px[a.f * 3]; y += sim.px[a.f * 3 + 1]; z += sim.px[a.f * 3 + 2]; }
            goal.set(x / m.length, y / m.length + 0.1, z / m.length);
        }
        this.focus.lerp(goal, Math.min(1, dt * 1.5));
        const idle = performance.now() - this.userAt > 6000 && !this.userHolding;
        if (idle) {
            const delta = goal.clone().sub(this.controls.target).multiplyScalar(Math.min(1, dt * 0.6));
            this.controls.target.add(delta);
            this.camera.position.add(delta);
        }
        this.sun.target.position.lerp(goal, Math.min(1, dt));
        this.sun.position.copy(this.sun.target.position).add(this.sunOffset);
    }

    render(dt) {
        if (!this.rtScene || !this.sim) return;
        this.follow(dt, this.sim);
        this.controls.update();
        const r = this.renderer;
        r.setRenderTarget(this.rtScene);
        r.render(this.scene, this.camera);
        const U = this.dof.mat.uniforms;
        U.tColor.value = this.rtScene.texture;
        U.tDepth.value = this.rtScene.depthTexture;
        U.near.value = this.camera.near; U.far.value = this.camera.far;
        // focus on the bridge: the distance from the camera to the focus point
        const fd = this.camera.position.distanceTo(this.focus);
        U.focus.value = fd;
        // macro lens: depth of field is a few millimetres at 15 cm
        U.focusScale.value = this.aperture * fd * 1.6;
        r.setRenderTarget(this.rtDof);
        r.render(this.dof.scene, this.postCam);
        const F = this.final.mat.uniforms;
        F.tColor.value = this.rtScene.texture;
        F.tBlur.value = this.rtDof.texture;
        F.tDepth.value = this.rtScene.depthTexture;
        F.near.value = this.camera.near; F.far.value = this.camera.far;
        F.focus.value = U.focus.value; F.focusScale.value = U.focusScale.value;
        F.time.value = this.time;
        r.setRenderTarget(null);
        r.render(this.final.scene, this.postCam);
    }
}

// Single-pass gather bokeh after Dennis Gustafsson ("Bokeh depth of field in a single pass").
// The loop is bounded; the radius grows every step.
const DOF_FRAG = /* glsl */`
#include <packing>
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform float near, far, focus, focusScale, maxBlur;
uniform vec2 pixel;
varying vec2 vUv;
const float GOLDEN = 2.39996323;
float viewDepth(vec2 uv) {
    float d = texture2D(tDepth, uv).x;
    return -perspectiveDepthToViewZ(d, near, far);
}
float blurSize(float depth) {
    float coc = clamp((1.0 / focus - 1.0 / depth) * focusScale, -1.0, 1.0);
    return abs(coc) * maxBlur;
}
void main() {
    float cd = viewDepth(vUv);
    float cs = blurSize(cd);
    vec3 col = texture2D(tColor, vUv).rgb;
    float tot = 1.0;
    // a random start per pixel turns the sampling pattern into fine noise
    float jit = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float radius = 0.5 + jit * 0.8;
    float ang = jit * 6.2831853;
    float spread = cs;   // how blurred this pixel ends up, including blurred foreground spilling over it
    for (int i = 0; i < 200; i++) {
        if (radius >= maxBlur) break;
        vec2 tc = vUv + vec2(cos(ang), sin(ang)) * pixel * radius;
        vec3 sc = texture2D(tColor, tc).rgb;
        float sd = viewDepth(tc);
        float ss = blurSize(sd);
        if (sd > cd) ss = clamp(ss, 0.0, cs * 2.0);
        float m = smoothstep(radius - 0.5, radius + 0.5, ss);
        if (sd < cd) spread = max(spread, ss * m);
        col += mix(col / tot, sc, m);
        tot += 1.0;
        ang += GOLDEN;
        radius += 1.4 / radius;
    }
    gl_FragColor = vec4(col / tot, spread / maxBlur);
}`;

const FINAL_FRAG = /* glsl */`
#include <packing>
uniform sampler2D tColor;
uniform sampler2D tBlur;
uniform sampler2D tDepth;
uniform float near, far, focus, focusScale, maxBlur;
uniform float exposure, time;
uniform vec2 resolution;
varying vec2 vUv;
// ACES filmic fit (Stephen Hill), as in three.js
vec3 rrtOdt(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
vec3 aces(vec3 c) {
    const mat3 inM = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
    const mat3 outM = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
    c *= 1.0 / 0.6;
    c = outM * rrtOdt(inM * c);
    return clamp(c, 0.0, 1.0);
}
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + time * 0.0) * 43758.5453); }
void main() {
    float depth = -perspectiveDepthToViewZ(texture2D(tDepth, vUv).x, near, far);
    float coc = abs(clamp((1.0 / focus - 1.0 / depth) * focusScale, -1.0, 1.0)) * maxBlur;
    vec4 blur = texture2D(tBlur, vUv);
    float w = smoothstep(0.6, 2.5, max(coc, blur.a * maxBlur));
    vec3 c = mix(texture2D(tColor, vUv).rgb, blur.rgb, w) * exposure;
    c = aces(c);
    // slight lift and warmth, like a field macro photo
    vec2 q = vUv - 0.5;
    float vig = smoothstep(0.85, 0.2, length(q * vec2(1.1, 1.0)));
    c *= mix(0.72, 1.0, vig);
    c = pow(c, vec3(1.0 / 2.2));
    c += (hash(vUv * resolution + fract(time) * 100.0) - 0.5) * 0.018;
    gl_FragColor = vec4(c, 1.0);
}`;
