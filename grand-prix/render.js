// three.js view: sky, sun, the circuit, two cars, and the cameras.
import * as THREE from 'three';
import { Sky } from 'three/addons/objects/Sky.js';
import { makeCar } from './car-model.js';
import { buildCircuit, makeTextures, centerline, racingLine, toWorld } from './scene.js';

const SUN = new THREE.Vector3().setFromSphericalCoords(1, THREE.MathUtils.degToRad(52), THREE.MathUtils.degToRad(140));

export class View {
    constructor(canvas, palette) {
        this.palette = palette;
        const r = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
        r.setPixelRatio(Math.min(devicePixelRatio, 2));
        r.toneMapping = THREE.NeutralToneMapping;
        r.toneMappingExposure = 0.72;
        r.shadowMap.enabled = true;
        r.shadowMap.type = THREE.PCFShadowMap;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(58, 1, 0.3, 9000);
        this.textures = makeTextures(palette);

        const sky = new Sky();
        sky.scale.setScalar(20000);
        const u = sky.material.uniforms;
        u.turbidity.value = 6; u.rayleigh.value = 1.4; u.mieCoefficient.value = 0.004; u.mieDirectionalG.value = 0.8;
        u.sunPosition.value.copy(SUN);
        this.scene.add(sky);
        const pmrem = new THREE.PMREMGenerator(r);
        const envScene = new THREE.Scene();
        envScene.add(sky.clone());
        this.scene.environment = pmrem.fromScene(envScene, 0.04).texture;
        this.scene.environmentIntensity = 0.3;
        this.scene.fog = new THREE.Fog(palette.haze, 900, 7000);

        this.scene.add(new THREE.HemisphereLight(0xdfe8f0, 0x4a5a3a, 0.7));
        const sun = this.sun = new THREE.DirectionalLight(0xfff1dd, 2.3);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        const sc = sun.shadow.camera;
        sc.left = sc.bottom = -70; sc.right = sc.top = 70; sc.near = 1; sc.far = 600;
        sun.shadow.bias = -0.0004;
        sun.shadow.normalBias = 0.03;
        this.scene.add(sun, sun.target);

        this.cars = {
            ai: makeCar(palette.ai),
            you: makeCar(palette.you),
        };
        this.scene.add(this.cars.ai, this.cars.you);
        this.mode = 'chase';
        this.chase = { yaw: 0, pos: new THREE.Vector3(), ready: false };
        this.tv = null;
        this.resize();
        addEventListener('resize', () => this.resize());
    }

    resize() {
        const w = innerWidth, h = innerHeight;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    setCircuit(track, centerData) {
        if (this.circuit) {
            this.scene.remove(this.circuit.world);
            this.circuit.world.traverse(o => { o.geometry?.dispose(); });
        }
        const c = centerline(centerData, track.center_count);
        this.center = c;
        this.circuit = buildCircuit(c, racingLine(track), this.palette, this.textures);
        this.scene.add(this.circuit.world);
        this.chase.ready = false;
    }

    setLights(n, color) { this.circuit?.setLights(n, color); }

    placeCar(name, state, prev, alpha, dt) {
        const car = this.cars[name];
        const lerp = (i) => prev ? prev[i] + (state[i] - prev[i]) * alpha : state[i];
        let yaw = state[2];
        if (prev) {
            let d = state[2] - prev[2];
            d = Math.atan2(Math.sin(d), Math.cos(d));
            yaw = prev[2] + d * alpha;
        }
        car.position.copy(toWorld(lerp(0), lerp(1), 0));
        car.rotation.set(0, yaw, 0);
        // a little pitch and roll from the accelerations, for weight
        car.userData.update(lerp(6), lerp(3), dt);
        return car;
    }

    ghost(on) {
        const car = this.cars.ai;
        if (car.userData.ghost === on) return;
        car.userData.ghost = on;
        car.traverse(o => {
            if (!o.isMesh) return;
            o.material = o.material.clone();
            o.material.transparent = on;
            o.material.opacity = on ? 0.38 : 1;
            o.material.depthWrite = !on;
        });
    }

    // follow: the car the cameras look at
    frame(race, prevStates, alpha, dt) {
        const you = this.placeCar('you', race.you.state, prevStates?.you, alpha, dt);
        const ai = this.placeCar('ai', race.ai.state, prevStates?.ai, alpha, dt);
        this.ghost(you.visible && you.position.distanceTo(ai.position) < 6.5);
        const target = this.follow === 'ai' ? ai : you;
        const state = this.follow === 'ai' ? race.ai.state : race.you.state;
        this.updateCamera(target, state, race, dt);
        this.sun.position.copy(target.position).addScaledVector(SUN, 300);
        this.sun.target.position.copy(target.position);
        this.renderer.render(this.scene, this.camera);
    }

    updateCamera(target, state, race, dt) {
        const cam = this.camera;
        const yaw = target.rotation.y;
        const fwd = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
        if (this.mode === 'onboard') {
            cam.fov = 70;
            cam.position.copy(target.position).addScaledVector(fwd, 0.05).add(new THREE.Vector3(0, 1.22, 0));
            cam.lookAt(target.position.clone().addScaledVector(fwd, 30).add(new THREE.Vector3(0, 0.6, 0)));
        } else if (this.mode === 'tv') {
            const cams = this.circuit.cameras;
            if (!this.tv || target.position.distanceTo(this.tv.pos) > 260 || this.tvBehind(target, fwd)) {
                let best = null, bd = Infinity;
                for (const c of cams) {
                    const d = c.pos.distanceTo(target.position);
                    const ahead = c.pos.clone().sub(target.position).dot(fwd);
                    if (ahead > -20 && d < bd) { bd = d; best = c; }
                }
                this.tv = best ?? cams[0];
            }
            const d = this.tv.pos.distanceTo(target.position);
            cam.fov = THREE.MathUtils.clamp(2 * Math.atan(9 / d) * 180 / Math.PI, 5, 55);
            cam.position.copy(this.tv.pos);
            cam.lookAt(target.position.clone().add(new THREE.Vector3(0, 0.6, 0)));
        } else {
            // chase: lag the heading a little so corners show the car turning
            const ch = this.chase;
            if (!ch.ready) { ch.yaw = yaw; ch.ready = true; }
            let d = yaw - ch.yaw;
            d = Math.atan2(Math.sin(d), Math.cos(d));
            ch.yaw += d * Math.min(1, dt * 6);
            const back = new THREE.Vector3(Math.cos(ch.yaw), 0, -Math.sin(ch.yaw));
            const speed = state[3];
            cam.fov = 56 + Math.min(12, speed / 8);
            cam.position.copy(target.position).addScaledVector(back, -8.2).add(new THREE.Vector3(0, 2.5, 0));
            cam.lookAt(target.position.clone().addScaledVector(back, 5).add(new THREE.Vector3(0, 0.9, 0)));
        }
        cam.updateProjectionMatrix();
    }

    tvBehind(target, fwd) {
        // switch when the car has gone well past the camera
        return this.tv.pos.clone().sub(target.position).dot(fwd) < -60;
    }

    // Slow orbit over the circuit while the track list is open.
    overview(t) {
        const b = this.circuit.bounds, cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
        const r = Math.max(b.maxX - b.minX, b.maxY - b.minY);
        const a = t * 0.04;
        this.camera.fov = 40;
        this.camera.position.copy(toWorld(cx + Math.cos(a) * r * 0.75, cy + Math.sin(a) * r * 0.75, r * 0.42));
        this.camera.lookAt(toWorld(cx, cy, 0));
        this.camera.updateProjectionMatrix();
        this.sun.position.copy(toWorld(cx, cy, 0)).addScaledVector(SUN, 300);
        this.sun.target.position.copy(toWorld(cx, cy, 0));
        this.renderer.render(this.scene, this.camera);
    }
}
