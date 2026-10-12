// Draws the race with three.js in the simulator's z-up world. The airfield is a
// photo panorama projected from the pilot's eye onto a ground disc and a distant
// dome: from the pilot's spot it is exactly the photo, and the chase and FPV
// cameras still see the runway pass beneath them. A sun light matched to the
// photo casts the drone's and the rings' shadows onto the photographed ground.
// Hedges, reeds and tree lines stand up as walls at their distance in the photo, so
// the field still holds together when the free camera moves away from the pilot.
import * as THREE from 'three';
import { EYE_HEIGHT } from './layout.js';

const FIELD_RADIUS = 220;                 // m, sky dome around the pilot
// Distance (m) from the photographer to the foot of the hedges, reeds and tree lines,
// read off the panorama by column (of 4096; 150 = the far tree line). The ground stops
// there and a wall carries the rest of the photo, with its sky cut out.
const PHOTO_WALLS = [[0, 150], [40, 150], [48, 63], [150, 63], [162, 150], [200, 150], [212, 28],
    [250, 17], [285, 19], [330, 23], [512, 27], [560, 40], [588, 150], [925, 150], [935, 99],
    [1110, 110], [1122, 150], [1700, 150], [1712, 56], [1760, 30], [1900, 30], [2000, 33],
    [2012, 99], [2180, 99], [2200, 63], [2240, 36], [2330, 45], [2420, 56], [2560, 60],
    [2800, 40], [3072, 33], [3584, 45], [3800, 50], [3822, 150], [4096, 150]];
const WALL_SEGMENTS = 720;
const BACKDROP = 190;                     // m, a last wall behind everything, for the far tree line
const WALL_TOP = Math.tan(60 * Math.PI / 180);   // walls reach 60° above the pilot's eye (or the dome)
const COURSE_CLEARANCE = 8;               // m, walls stand at least this far behind any gate
const PHOTO_SUN = { u: 0.6035, elevation: 18.6 * Math.PI / 180 };  // where the sun is in the panorama
const PHOTO_VIEW_U = 330 / 360;           // panorama column the pilot faces (down the runway)
const RING_TUBE = 0.075;                  // tube radius, as in the collision model
const ORANGE = '#ff4808', WHITE = '#f5f5eb';  // the RealFlight tube ring's safety bands
const CHASE = { distance: Math.hypot(1.3, 0.42), elevation: Math.atan2(0.42, 1.3) };  // 1.3 m behind, 0.42 m up
// The photo holds together up to about spectator height; above that its grass smears
const FREE = { fov: 50, minDistance: 0.6, maxDistance: 60, maxHeight: 6, minCameraHeight: 0.12,
    maxCameraHeight: 6, wallMargin: 2 };
const Z = new THREE.Vector3(0, 0, 1);
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

// Camera position for an orbit { target, azimuth, elevation, distance } (z-up)
function orbitPosition(o, out) {
    const c = Math.cos(o.elevation);
    return out.set(c * Math.cos(o.azimuth), c * Math.sin(o.azimuth), Math.sin(o.elevation))
        .multiplyScalar(o.distance).add(o.target);
}
const wrapAngle = a => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

const fieldVertex = /* glsl */`
    varying vec3 vWorld;
    void main() {
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
    }`;
// Look up the panorama in the direction from the photographer's eye
const fieldFragment = /* glsl */`
    uniform sampler2D map;
    uniform vec3 eye;
    uniform float yaw;
    varying vec3 vWorld;
    void main() {
        vec3 d = normalize(vWorld - eye);
        float u = (yaw - atan(d.y, d.x)) / 6.28318530718;
        vec2 uv = vec2(fract(u), 0.5 + asin(clamp(d.z, -1.0, 1.0)) / 3.14159265359);
        // Gradients from a copy of u that is continuous across the seam, so no mip seam shows
        float ua = fract(u + 0.5) - 0.5;
        vec2 gx = vec2(abs(dFdx(uv.x)) < abs(dFdx(ua)) ? dFdx(uv.x) : dFdx(ua), dFdx(uv.y));
        vec2 gy = vec2(abs(dFdy(uv.x)) < abs(dFdy(ua)) ? dFdy(uv.x) : dFdy(ua), dFdy(uv.y));
        // Seen from away from the pilot, a photo pixel of ground at distance r is r / eye height
        // times longer than it is wide, so the grass smears into streaks pointing at the pilot.
        // Blur it sideways to match there (the pilot's own view is untouched).
        if (vWorld.z < 0.01) {
            float away = min(distance(cameraPosition, eye) / 8.0, 1.0);
            float r = length(vWorld.xy - eye.xy);
            float blur = away * min(0.7 * r / eye.z, 48.0) / float(textureSize(map, 0).x);
            gx.x = sign(gx.x) * max(abs(gx.x), blur);
        }
        gl_FragColor = textureGrad(map, uv, gx, gy);
        #ifdef CUTOUT
        // walls: let the sky through to the dome behind (same photo pixel from the pilot's eye)
        vec3 c = gl_FragColor.rgb;
        float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
        if ((c.b > c.g + 0.015 && lum > 0.42) || min(c.r, min(c.g, c.b)) > 0.72) discard;
        #endif
        #include <colorspace_fragment>
    }`;

function stripes() {
    // Two white bands and two orange arcs around the ring, as in the RealFlight asset's texture
    const c = document.createElement('canvas');
    c.width = 256; c.height = 4;
    const g = c.getContext('2d');
    for (let k = 0; k < 2; k++) {
        g.fillStyle = WHITE; g.fillRect(k * 128, 0, 32, 4);
        g.fillStyle = ORANGE; g.fillRect(k * 128 + 32, 0, 96, 4);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = THREE.RepeatWrapping;
    t.anisotropy = 4;
    return t;
}

// A 5" racing quad on the Tempest's rotor layout (arms at ±39.4°, rotors 0.0747 m
// forward/back and 0.0613 m to the side), colored after a Commons photo of one.
function buildDrone(sites) {
    const drone = new THREE.Group();
    const carbon = new THREE.MeshStandardMaterial({ color: 0x14110f, roughness: 0.42, metalness: 0.25 });
    const black = new THREE.MeshStandardMaterial({ color: 0x0c0b0b, roughness: 0.6 });
    const battery = new THREE.MeshStandardMaterial({ color: 0x3db47a, roughness: 0.5 });
    const label = new THREE.MeshStandardMaterial({ color: 0xe9eef0, roughness: 0.6 });
    const metal = new THREE.MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.3, metalness: 0.9 });
    const bell = new THREE.MeshStandardMaterial({ color: 0x8c2a1c, roughness: 0.35, metalness: 0.7 });
    const box = (sx, sy, sz, m, x = 0, y = 0, z = 0) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), m);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        drone.add(mesh);
        return mesh;
    };
    const arm = Math.hypot(sites[0][0], sites[0][1]) * 2 + 0.03;
    for (const s of [1, -1]) box(arm, 0.017, 0.004, carbon).rotation.z = s * Math.atan2(sites[0][1], sites[0][0]);
    box(0.105, 0.044, 0.003, carbon, 0, 0, 0.004);
    box(0.088, 0.04, 0.0025, carbon, -0.004, 0, 0.034);
    box(0.032, 0.032, 0.012, new THREE.MeshStandardMaterial({ color: 0x23303a, roughness: 0.7 }), -0.005, 0, 0.016);
    for (const x of [-0.035, 0.035]) for (const y of [-0.017, 0.017]) {
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.0025, 0.0025, 0.03, 8), metal);
        post.rotation.x = Math.PI / 2;
        post.position.set(x, y, 0.019);
        drone.add(post);
    }
    box(0.072, 0.035, 0.027, battery, -0.008, 0, 0.049);
    box(0.03, 0.0352, 0.012, label, -0.02, 0, 0.051);
    box(0.012, 0.037, 0.029, black, 0.006, 0, 0.049);
    // FPV camera, tilted up 25° as racers mount them
    const cam = new THREE.Group();
    cam.position.set(0.052, 0, 0.02);
    cam.rotation.y = -25 * Math.PI / 180;
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.02, 0.021, 0.02), black);
    body.castShadow = true;
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.0075, 0.0075, 0.012, 16), black);
    lens.rotation.z = Math.PI / 2;
    lens.position.x = 0.014;
    const glass = new THREE.Mesh(new THREE.CircleGeometry(0.0055, 16),
        new THREE.MeshStandardMaterial({ color: 0x223355, roughness: 0.05, metalness: 0.8 }));
    glass.rotation.y = Math.PI / 2;
    glass.position.x = 0.0202;
    cam.add(body, lens, glass);
    drone.add(cam);
    // Video antenna, swept back
    const ant = new THREE.Mesh(new THREE.CylinderGeometry(0.0018, 0.0018, 0.055, 6),
        new THREE.MeshStandardMaterial({ color: 0xaa3322, roughness: 0.6 }));
    ant.position.set(-0.058, 0, 0.052);
    ant.rotation.y = -0.6;
    ant.rotation.x = Math.PI / 2;
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.009, 12, 8), black);
    cap.scale.set(1, 1, 0.55);
    cap.position.set(-0.074, 0, 0.075);
    drone.add(ant, cap);

    // Motors and props
    const props = [];
    const propColor = 0xff8f3a;
    const bladeShape = new THREE.Shape();
    bladeShape.moveTo(0.006, -0.004);
    bladeShape.quadraticCurveTo(0.03, -0.011, 0.0635, -0.004);
    bladeShape.quadraticCurveTo(0.065, 0.002, 0.06, 0.004);
    bladeShape.quadraticCurveTo(0.03, 0.006, 0.006, 0.004);
    const bladeGeom = new THREE.ShapeGeometry(bladeShape, 6);
    const discGeom = new THREE.RingGeometry(0.008, 0.0635, 40);
    sites.forEach((site, i) => {
        const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.0135, 0.0135, 0.016, 20), bell);
        motor.rotation.x = Math.PI / 2;
        motor.position.set(site[0], site[1], 0.01);
        motor.castShadow = true;
        drone.add(motor);
        const hub = new THREE.Group();
        hub.position.set(site[0], site[1], 0.0215);
        const bladeMat = new THREE.MeshStandardMaterial({ color: propColor, roughness: 0.5, side: THREE.DoubleSide, transparent: true });
        const blades = new THREE.Group();
        for (let k = 0; k < 3; k++) {
            const b = new THREE.Mesh(bladeGeom, bladeMat);
            b.rotation.z = k * 2 * Math.PI / 3;
            b.castShadow = true;
            blades.add(b);
        }
        const disc = new THREE.Mesh(discGeom, new THREE.MeshBasicMaterial({
            color: propColor, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
        hub.add(blades, disc);
        drone.add(hub);
        props.push({ blades, bladeMat, disc, angle: Math.random() * 6, spin: i % 2 ? 1 : -1 });
    });
    return { drone, props };
}

export class View {
    constructor(canvas) {
        this.canvas = canvas;
        const renderer = this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.NeutralToneMapping;
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFShadowMap;
        this.maxTexture = renderer.capabilities.maxTextureSize;

        const scene = this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.02, 2000);
        this.camera.up.set(0, 0, 1);
        this.mode = 'pilot';

        // Photo field
        this.fieldMaterial = new THREE.ShaderMaterial({
            uniforms: { map: { value: null }, eye: { value: new THREE.Vector3() }, yaw: { value: 0 } },
            vertexShader: fieldVertex, fragmentShader: fieldFragment,
            side: THREE.DoubleSide, toneMapped: false, depthWrite: true,
        });
        this.wallMaterial = new THREE.ShaderMaterial({
            uniforms: this.fieldMaterial.uniforms, defines: { CUTOUT: '' },
            vertexShader: fieldVertex, fragmentShader: fieldFragment,
            side: THREE.DoubleSide, toneMapped: false,
        });
        // Ground out to the backdrop; the walls stand on it
        this.ground = new THREE.Mesh(new THREE.CircleGeometry(BACKDROP, 160), this.fieldMaterial);
        this.walls = new THREE.Mesh(new THREE.BufferGeometry(), this.wallMaterial);
        this.wallDistance = new Float32Array(WALL_SEGMENTS);
        const domeGeom = new THREE.SphereGeometry(FIELD_RADIUS, 160, 48, 0, Math.PI * 2, 0, Math.PI / 2);
        domeGeom.rotateX(Math.PI / 2);   // pole to +z
        this.dome = new THREE.Mesh(domeGeom, this.fieldMaterial);
        this.field = new THREE.Group();
        this.field.add(this.ground, this.walls, this.dome);
        this.field.visible = false;
        scene.add(this.field);

        // Shadows land on the photographed ground
        this.shadowCatcher = new THREE.Mesh(new THREE.PlaneGeometry(140, 140),
            new THREE.ShadowMaterial({ opacity: 0.5, color: 0x10141c }));
        this.shadowCatcher.position.z = 0.003;
        this.shadowCatcher.receiveShadow = true;
        scene.add(this.shadowCatcher);

        const sun = this.sun = new THREE.DirectionalLight(0xfff1dc, 2.6);
        sun.castShadow = true;
        sun.shadow.mapSize.set(4096, 4096);
        const sc = sun.shadow.camera;
        sc.left = -24; sc.right = 24; sc.top = 24; sc.bottom = -24; sc.near = 1; sc.far = 200;
        sun.shadow.bias = -0.0002;
        sun.shadow.normalBias = 0.01;
        scene.add(sun, sun.target);
        // The drone is too small for the course-wide shadow map, so a second sun (adding
        // no light worth mentioning) carries a tight shadow map that follows it
        const near = this.droneSun = new THREE.DirectionalLight(0xffffff, 0.0001);
        near.castShadow = true;
        near.shadow.mapSize.set(1024, 1024);
        const nc = near.shadow.camera;
        nc.left = -0.6; nc.right = 0.6; nc.top = 0.6; nc.bottom = -0.6; nc.near = 0.5; nc.far = 80;
        near.shadow.bias = -0.0005;
        scene.add(near, near.target);
        this.hemi = new THREE.HemisphereLight(0xbcd2ec, 0x4a4a3c, 0.35);
        this.hemi.position.set(0, 0, 1);
        scene.add(this.hemi);

        this.course = new THREE.Group();
        scene.add(this.course);
        this.ringTexture = stripes();

        // Launch pad: a dark mat with an orange border
        const padGroup = this.pad = new THREE.Group();
        const mat = new THREE.Mesh(new THREE.PlaneGeometry(0.7, 0.7), new THREE.MeshStandardMaterial({ color: 0x1c1d20, roughness: 0.9 }));
        const border = new THREE.Mesh(new THREE.RingGeometry(0.33, 0.35, 4, 1, Math.PI / 4),
            new THREE.MeshStandardMaterial({ color: ORANGE, roughness: 0.8 }));
        mat.position.z = 0.004; border.position.z = 0.005;
        mat.receiveShadow = true;
        padGroup.add(mat, border);
        scene.add(padGroup);

        this.lookAt = new THREE.Vector3();
        this.lookVel = new THREE.Vector3();
        this.fov = 50;
        this.chasePos = null;
        // Chase: an orbit around the drone, relative to straight behind it
        this.chaseGoal = { azimuth: 0, elevation: CHASE.elevation, distance: CHASE.distance };
        this.chaseNow = { ...this.chaseGoal };
        // Free: an orbit around a point on the field; inputs move the goal, the camera eases to it
        this.freeGoal = { target: new THREE.Vector3(), azimuth: 0, elevation: 0.45, distance: 30 };
        this.freeNow = { target: new THREE.Vector3(), azimuth: 0, elevation: 0.45, distance: 30 };
        this.freeUp = new THREE.Vector3(0, 0, 1);
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

    // The panorama: a quick 4K version first, then 8K where the GPU takes it
    async loadField(base) {
        const loader = new THREE.TextureLoader();
        const load = url => new Promise((ok, fail) => loader.load(url, ok, undefined, fail));
        const use = tex => {
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.wrapS = THREE.RepeatWrapping;
            tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
            const old = this.fieldMaterial.uniforms.map.value;
            this.fieldMaterial.uniforms.map.value = tex;
            this.field.visible = true;
            this.updateEnvironment();
            if (old) old.dispose();
        };
        use(await load(base + 'field-4k.jpg'));
        if (this.maxTexture >= 8192) load(base + 'field-8k.jpg').then(use).catch(() => {});
    }

    // Image-based light for the drone and rings, rendered from the photo field itself
    updateEnvironment() {
        if (!this.layout || !this.fieldMaterial.uniforms.map.value) return;
        const pmrem = this.pmrem ??= new THREE.PMREMGenerator(this.renderer);
        const envScene = new THREE.Scene();
        envScene.add(new THREE.Mesh(this.ground.geometry, this.fieldMaterial), new THREE.Mesh(this.walls.geometry, this.wallMaterial),
            new THREE.Mesh(this.dome.geometry, this.fieldMaterial));
        envScene.children.forEach(m => m.position.copy(this.field.position));
        const c = this.layout.center;
        const target = pmrem.fromScene(envScene, 0.02, 0.1, FIELD_RADIUS * 3, { position: new THREE.Vector3(c[0], c[1], 3) });
        if (this.scene.environment) this.scene.environment.dispose();
        this.scene.environment = target.texture;
        this.scene.environmentIntensity = 0.9;
    }

    // Place the field, pilot and lights for a course, and build its rings
    setCourse(flight, layout) {
        this.layout = layout;
        const [px, py] = layout.pilot;
        this.eye = new THREE.Vector3(px, py, EYE_HEIGHT);
        this.field.position.set(px, py, 0);
        // Turn the panorama so the pilot faces down the runway
        const yaw = layout.heading + 2 * Math.PI * PHOTO_VIEW_U;
        this.fieldMaterial.uniforms.eye.value.copy(this.eye);
        this.fieldMaterial.uniforms.yaw.value = yaw;
        const sunLon = yaw - 2 * Math.PI * PHOTO_SUN.u, el = PHOTO_SUN.elevation;
        const sunDir = this.sunDir = new THREE.Vector3(Math.cos(el) * Math.cos(sunLon), Math.cos(el) * Math.sin(sunLon), Math.sin(el));
        const [cx, cy] = layout.center;
        this.sun.target.position.set(cx, cy, 0);
        this.sun.position.set(cx, cy, 0).addScaledVector(sunDir, 80);
        this.shadowCatcher.position.set(cx, cy, 0.003);
        const c = flight.course;
        this.buildField(yaw, [...c.centers, flight.pad]);

        this.course.clear();
        let reach = 0, height = 0;
        for (let i = 0; i < c.count; i++) {
            reach = Math.max(reach, Math.hypot(c.centers[i][0] - cx, c.centers[i][1] - cy));
            height += c.centers[i][2] / c.count;
        }
        this.courseSize = { reach, height };
        const ringMat = new THREE.MeshStandardMaterial({ map: this.ringTexture, roughness: 0.55, metalness: 0.0 });
        for (let i = 0; i < c.count; i++) {
            const major = 0.5 * (c.inner_radius[i] + c.outer_radius[i]);
            const ring = new THREE.Mesh(new THREE.TorusGeometry(major, RING_TUBE, 20, 120), ringMat);
            const r = new THREE.Vector3(...c.right[i]), u = new THREE.Vector3(...c.up[i]), n = new THREE.Vector3(...c.normals[i]);
            ring.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(r, u, n));
            ring.position.set(...c.centers[i]);
            ring.castShadow = true;
            this.course.add(ring);
        }
        this.pad.position.set(flight.pad[0], flight.pad[1], 0);
        const n0 = c.normals[0];
        this.pad.rotation.z = Math.atan2(n0[1], n0[0]);

        if (!this.drone) {
            const { drone, props } = buildDrone(flight.track.plant.rotor_sites);
            this.drone = drone; this.props = props;
            this.scene.add(drone);
        }
        this.lookAt.set(flight.pad[0], flight.pad[1], 0.3);
        this.lookVel.set(0, 0, 0);
        this.chasePos = null;
        this.updateEnvironment();
        if (this.mode === 'free') this.freeOverview(false);
    }

    // Walls around the pilot (the field group's origin), and the backdrop behind them
    buildField(yaw, keep) {
        const n = WALL_SEGMENTS, D = this.wallDistance, [px, py] = this.layout.pilot;
        for (let i = 0; i < n; i++) {
            const u = (((yaw - 2 * Math.PI * i / n) / (2 * Math.PI)) % 1 + 1) % 1 * 4096;
            let k = 1;
            while (PHOTO_WALLS[k][0] < u) k++;
            const [u0, d0] = PHOTO_WALLS[k - 1], [u1, d1] = PHOTO_WALLS[k];
            D[i] = d0 + (d1 - d0) * (u - u0) / (u1 - u0);
        }
        // Push walls back behind the course where the photo's hedges would cut through it
        for (const [x, y] of keep) {
            const r = Math.hypot(x - px, y - py), a = Math.atan2(y - py, x - px);
            const half = Math.atan2(COURSE_CLEARANCE, r);
            for (let j = Math.floor((a - half) / (2 * Math.PI) * n); j <= Math.ceil((a + half) / (2 * Math.PI) * n); j++) {
                const i = ((j % n) + n) % n;
                D[i] = Math.min(Math.max(D[i], r + COURSE_CLEARANCE), BACKDROP - 10);
            }
        }
        const wall = [], index = [];
        const ring = (dist, skipJumps) => {
            const base = wall.length / 3;
            for (let i = 0; i <= n; i++) {
                const a = 2 * Math.PI * i / n, d = dist(i % n), x = d * Math.cos(a), y = d * Math.sin(a);
                wall.push(x, y, 0, x, y, Math.min(EYE_HEIGHT + d * WALL_TOP, Math.sqrt(FIELD_RADIUS ** 2 - d * d) - 1));
                // no wall across a jump in distance: from the side it would be one pixel column stretched
                if (i < n && !(skipJumps && Math.abs(dist((i + 1) % n) - d) > 3)) {
                    const k = base + 2 * i;
                    index.push(k, k + 2, k + 1, k + 1, k + 2, k + 3);
                }
            }
        };
        ring(i => D[i], true);
        ring(() => BACKDROP, false);
        this.walls.geometry.dispose();
        const g = this.walls.geometry = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(wall, 3));
        g.setIndex(index);
        g.computeBoundingSphere();
    }

    // How far (m) a point is inside the walls
    insideWalls(x, y) {
        const [px, py] = this.layout.pilot, n = WALL_SEGMENTS;
        const a = Math.atan2(y - py, x - px), f = ((a / (2 * Math.PI) * n) % n + n) % n;
        const i = Math.floor(f), t = f - i, D = this.wallDistance;
        return D[i] + (D[(i + 1) % n] - D[i]) * t - Math.hypot(x - px, y - py);
    }

    // keepCamera: start the free camera from the current view instead of the course overview
    setMode(mode, keepCamera = false) {
        if (mode === 'free') {
            if (keepCamera) this.freeFromCamera();
            else this.freeOverview(this.mode !== 'free' && !reducedMotion.matches);
        }
        if (mode === 'chase' && this.mode !== 'chase') {
            Object.assign(this.chaseGoal, { azimuth: 0, elevation: CHASE.elevation, distance: CHASE.distance });
            Object.assign(this.chaseNow, this.chaseGoal);
            this.chasePos = null;
        }
        this.mode = mode;
    }

    // Free camera orbiting the point straight ahead, at the drone's distance
    freeFromCamera() {
        const cam = this.camera, ahead = new THREE.Vector3(0, 0, -1).applyQuaternion(cam.quaternion);
        const d = THREE.MathUtils.clamp(ahead.dot(this.drone.position.clone().sub(cam.position)), 2, FREE.maxDistance);
        const o = this.freeNow;
        o.target.copy(cam.position).addScaledVector(ahead, d);
        const v = cam.position.clone().sub(o.target);
        o.distance = d;
        o.elevation = Math.asin(THREE.MathUtils.clamp(v.z / d, -1, 1));
        o.azimuth = Math.atan2(v.y, v.x);
        this.freeGoal.target.copy(o.target);
        Object.assign(this.freeGoal, { azimuth: o.azimuth, elevation: o.elevation, distance: o.distance });
        this.clampFree();
        this.freeUp.set(0, 1, 0).applyQuaternion(cam.quaternion);
        this.fov = cam.fov;
    }

    // The whole course from above the pilot's side; `sweep` flies there from the current view
    freeOverview(sweep) {
        if (sweep) this.freeFromCamera();
        const [cx, cy] = this.layout.center, g = this.freeGoal;
        g.target.set(cx, cy, Math.min(this.courseSize.height, 3));
        g.distance = THREE.MathUtils.clamp(this.courseSize.reach * 2 + 6, 12, 55);
        g.elevation = Math.asin((FREE.maxCameraHeight - 1 - g.target.z) / g.distance);
        g.azimuth = this.layout.heading + Math.PI + 0.35;
        this.clampFree();
        if (sweep) g.azimuth = this.freeNow.azimuth + wrapAngle(g.azimuth - this.freeNow.azimuth);
        else {
            this.freeNow.target.copy(g.target);
            Object.assign(this.freeNow, { azimuth: g.azimuth, elevation: g.elevation, distance: g.distance });
            this.freeUp.copy(Z);
            this.fov = FREE.fov;
        }
    }

    // Keep the free camera's point and the camera inside the walls and above the ground
    clampFree() {
        const g = this.freeGoal, t = g.target, [px, py] = this.layout.pilot;
        const over = FREE.wallMargin + 2 - this.insideWalls(t.x, t.y);
        if (over > 0) {
            const r = Math.hypot(t.x - px, t.y - py), k = Math.max(r - over, 0) / r;
            t.x = px + (t.x - px) * k; t.y = py + (t.y - py) * k;
        }
        t.z = THREE.MathUtils.clamp(t.z, 0, FREE.maxHeight);
        g.distance = THREE.MathUtils.clamp(g.distance, FREE.minDistance, FREE.maxDistance);
        const low = Math.asin(THREE.MathUtils.clamp((FREE.minCameraHeight - t.z) / g.distance, -1, 1));
        const high = Math.asin(THREE.MathUtils.clamp((FREE.maxCameraHeight - t.z) / g.distance, -1, 1));
        g.elevation = THREE.MathUtils.clamp(g.elevation, Math.max(low, -1.3), Math.min(high, 1.5));
        const cam = this.scratch ??= new THREE.Vector3();
        for (let i = 0; i < 20 && this.insideWalls(orbitPosition(g, cam).x, cam.y) < FREE.wallMargin; i++) {
            g.distance = Math.max(g.distance * 0.85, FREE.minDistance);
        }
    }

    // Drag: swing around the drone (chase) or the free camera's point
    orbit(dx, dy) {
        const k = 1.2 * Math.PI / Math.max(this.canvas.clientHeight, 1);
        const o = this.mode === 'chase' ? this.chaseGoal : this.freeGoal;
        o.azimuth -= dx * k;
        o.elevation += dy * k;
        if (this.mode === 'chase') o.elevation = THREE.MathUtils.clamp(o.elevation, -0.35, 1.45);
        else this.clampFree();
    }

    // Right-drag or two fingers: drag the field along the ground
    pan(dx, dy) {
        const g = this.freeGoal, h = Math.max(this.canvas.clientHeight, 1);
        const s = g.distance * 2 * Math.tan(this.camera.fov * Math.PI / 360) / h;
        const along = s / Math.max(Math.sin(g.elevation), 0.35);
        const sa = Math.sin(g.azimuth), ca = Math.cos(g.azimuth);
        g.target.x += sa * dx * s - ca * dy * along;
        g.target.y += -ca * dx * s - sa * dy * along;
        this.clampFree();
    }

    zoom(factor) {
        if (this.mode === 'chase') {
            this.chaseGoal.distance = THREE.MathUtils.clamp(this.chaseGoal.distance * factor, 0.45, 8);
        } else {
            this.freeGoal.distance *= factor;
            this.clampFree();
        }
    }

    // Keys: move [right, forward, up] in -1..1; faster when zoomed out
    fly(move, dt) {
        if (!move[0] && !move[1] && !move[2]) return;
        const g = this.freeGoal, speed = THREE.MathUtils.clamp(g.distance * 0.8, 2.5, 18) * dt;
        const sa = Math.sin(g.azimuth), ca = Math.cos(g.azimuth);
        g.target.x += (-sa * move[0] - ca * move[1]) * speed;
        g.target.y += (ca * move[0] - sa * move[1]) * speed;
        g.target.z += move[2] * speed * 0.6;
        this.clampFree();
    }

    // pose: { p, q (w,x,y,z), v, thrust[4] (N), hover (N per rotor) }; dt is simulated
    // time (slower in slow motion), realDt wall-clock time for the viewer's camera moves
    draw(pose, dt, realDt = dt) {
        const { drone, camera } = this;
        drone.position.set(pose.p[0], pose.p[1], pose.p[2]);
        drone.quaternion.set(pose.q[1], pose.q[2], pose.q[3], pose.q[0]);
        // Props: blades blur into discs as they spin up
        this.props.forEach((prop, i) => {
            const spin = Math.sqrt(Math.max(pose.thrust[i], 0) / pose.hover);  // ~rpm / hover rpm
            prop.angle += prop.spin * Math.min(spin * 55, 40) * dt;
            prop.blades.rotation.z = prop.angle;
            const blur = Math.min(1, spin * 1.6);
            prop.bladeMat.opacity = 1 - 0.85 * blur;
            prop.disc.material.opacity = (this.mode === 'fpv' ? 0.1 : 0.22) * blur;
        });

        const p = drone.position;
        this.droneSun.target.position.copy(p);
        this.droneSun.position.copy(p).addScaledVector(this.sunDir, 20);
        if (this.mode === 'pilot') {
            // The pilot's head follows the drone (a stiff spring that also tracks its
            // velocity, so it stays centered); the lens zooms with distance
            const k = 40, damp = 2 * Math.sqrt(k), v = pose.v;
            const acc = new THREE.Vector3().subVectors(p, this.lookAt).multiplyScalar(k)
                .add(new THREE.Vector3(v[0], v[1], v[2]).sub(this.lookVel).multiplyScalar(damp));
            this.lookVel.addScaledVector(acc, dt);
            this.lookAt.addScaledVector(this.lookVel, dt);
            if (this.lookAt.distanceTo(p) > 4) { this.lookAt.copy(p); this.lookVel.set(...v); }
            camera.position.copy(this.eye);
            camera.up.set(0, 0, 1);
            camera.lookAt(this.lookAt);
            const dist = this.eye.distanceTo(this.lookAt);
            const want = THREE.MathUtils.clamp(2 * Math.atan(7 / 2 / dist) * 180 / Math.PI, 20, 60);
            this.fov += (want - this.fov) * Math.min(1, dt * 3);
            camera.fov = this.fov;
        } else if (this.mode === 'chase') {
            const fwd = new THREE.Vector3(pose.v[0], pose.v[1], 0);
            if (fwd.length() < 1) fwd.set(1, 0, 0).applyQuaternion(drone.quaternion).setZ(0);
            fwd.normalize();
            const o = this.chaseNow, k = 1 - Math.exp(-realDt * 12);
            for (const key of ['azimuth', 'elevation', 'distance']) o[key] += (this.chaseGoal[key] - o[key]) * k;
            const az = Math.atan2(fwd.y, fwd.x) + Math.PI + o.azimuth, ce = Math.cos(o.elevation);
            const want = new THREE.Vector3(ce * Math.cos(az), ce * Math.sin(az), Math.sin(o.elevation))
                .multiplyScalar(o.distance).add(p);
            want.z = Math.max(want.z, 0.25);
            if (!this.chasePos) this.chasePos = want.clone();
            this.chasePos.lerp(want, 1 - Math.exp(-dt * 6));
            camera.position.copy(this.chasePos);
            camera.up.set(0, 0, 1);
            // look a little ahead of the drone when behind it, straight at it from the side
            camera.lookAt(p.clone().addScaledVector(fwd, 1.8 * Math.max(Math.cos(o.azimuth), 0)));
            camera.fov = 75;
        } else if (this.mode === 'free') {
            const o = this.freeNow, g = this.freeGoal, k = 1 - Math.exp(-realDt * 10);
            o.target.lerp(g.target, k);
            for (const key of ['azimuth', 'elevation', 'distance']) o[key] += (g[key] - o[key]) * k;
            orbitPosition(o, camera.position);
            camera.position.z = Math.max(camera.position.z, 0.08);
            camera.up.copy(this.freeUp.lerp(Z, 1 - Math.exp(-realDt * 6)).normalize());
            camera.lookAt(o.target);
            this.fov += (FREE.fov - this.fov) * (1 - Math.exp(-realDt * 4));
            camera.fov = this.fov;
        } else {
            // FPV: the drone's camera, 25° up-tilt
            const offset = new THREE.Vector3(0.08, 0, 0.03).applyQuaternion(drone.quaternion);
            camera.position.copy(p).add(offset);
            const tilt = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -25 * Math.PI / 180);
            // camera looks down its -z with +y up; in the body frame that is forward (+x) with up (+z)
            const toBody = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
                new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
            camera.quaternion.copy(drone.quaternion).multiply(tilt).multiply(toBody);
            camera.fov = 88;
        }
        camera.updateProjectionMatrix();
        this.renderer.render(this.scene, camera);
    }
}
