// A procedural open-wheel race car. Local frame: +x forward, +y up, +z right;
// the origin is the center of mass (front axle 2.02 m ahead, rear 1.70 m
// behind, as in the simulation). Built from tapered boxes so it stays small.
import * as THREE from 'three';

const LF = 2.0215, LR = 1.7045, R_WHEEL = 0.36;

// A hexahedron spanning x0..x1, with width and bottom/top heights at each end.
function taper(x0, x1, w0, w1, b0, t0, b1, t1, zc = 0) {
    const v = [
        [x0, b0, zc - w0 / 2], [x0, b0, zc + w0 / 2], [x0, t0, zc + w0 / 2], [x0, t0, zc - w0 / 2],
        [x1, b1, zc - w1 / 2], [x1, b1, zc + w1 / 2], [x1, t1, zc + w1 / 2], [x1, t1, zc - w1 / 2],
    ];
    const faces = [[0, 1, 2, 3], [5, 4, 7, 6], [4, 0, 3, 7], [1, 5, 6, 2], [3, 2, 6, 7], [4, 5, 1, 0]];
    const pos = [];
    for (const [a, b, c, d] of faces) pos.push(...v[a], ...v[c], ...v[b], ...v[a], ...v[d], ...v[c]);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
}

function box(x0, x1, y0, y1, z0, z1) {
    const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
    g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    return g;
}

function wheel(width, tyre, rim, stripe) {
    const g = new THREE.Group();
    const t = new THREE.Mesh(new THREE.CylinderGeometry(R_WHEEL, R_WHEEL, width, 28, 1), tyre);
    t.rotation.x = Math.PI / 2;
    g.add(t);
    for (const side of [-1, 1]) {
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.2, 0.235, 28), stripe);
        ring.position.z = side * (width / 2 + 0.002);
        if (side < 0) ring.rotation.y = Math.PI;
        g.add(ring);
        const disc = new THREE.Mesh(new THREE.CircleGeometry(0.19, 20), rim);
        disc.position.z = side * (width / 2 - 0.04);
        if (side < 0) disc.rotation.y = Math.PI;
        g.add(disc);
    }
    // two spokes so rotation is visible
    const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.05, width - 0.06), rim);
    g.add(spoke);
    const spoke2 = spoke.clone();
    spoke2.rotation.z = Math.PI / 2;
    g.add(spoke2);
    return g;
}

// livery: { paint, accent, number }
export function makeCar(livery) {
    const car = new THREE.Group();
    const paint = new THREE.MeshPhysicalMaterial({ color: livery.paint, roughness: 0.35, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.12, envMapIntensity: 0.6 });
    const accent = new THREE.MeshPhysicalMaterial({ color: livery.accent, roughness: 0.3, metalness: 0.35, clearcoat: 1, clearcoatRoughness: 0.1 });
    const carbon = new THREE.MeshStandardMaterial({ color: 0x1b1c1e, roughness: 0.55, metalness: 0.2 });
    const tyre = new THREE.MeshStandardMaterial({ color: 0x161616, roughness: 0.9 });
    const rim = new THREE.MeshStandardMaterial({ color: 0x2a2b2d, roughness: 0.35, metalness: 0.8 });
    const stripe = new THREE.MeshStandardMaterial({ color: livery.tyreStripe ?? 0xf3f0e6, roughness: 0.7, side: THREE.DoubleSide });
    const helmetMat = new THREE.MeshPhysicalMaterial({ color: livery.helmet ?? livery.accent, roughness: 0.2, clearcoat: 1 });
    const add = (geo, mat) => { const m = new THREE.Mesh(geo, mat); m.castShadow = true; car.add(m); return m; };

    const y0 = 0.04;  // ride height of the floor
    // floor and diffuser
    add(taper(-2.0, 1.25, 1.45, 1.55, y0, y0 + 0.05, y0, y0 + 0.05), carbon);
    add(taper(-2.25, -1.85, 1.0, 1.1, y0 + 0.02, y0 + 0.32, y0, y0 + 0.06), carbon);
    // survival cell and nose
    add(taper(-0.35, 1.55, 0.72, 0.56, y0 + 0.08, 0.72, y0 + 0.1, 0.62), paint);
    add(taper(1.55, 3.05, 0.56, 0.2, y0 + 0.1, 0.62, y0 + 0.16, 0.32), paint);
    add(taper(1.4, 3.0, 0.08, 0.05, 0.6, 0.64, 0.3, 0.335), accent);  // nose stripe
    // sidepods (narrowing toward the gearbox: the "coke bottle")
    for (const s of [-1, 1]) {
        add(taper(-1.1, 0.75, 0.32, 0.42, y0 + 0.05, 0.42, y0 + 0.05, 0.56, s * 0.55), paint);
        add(taper(0.75, 0.95, 0.42, 0.36, y0 + 0.05, 0.56, y0 + 0.1, 0.5, s * 0.56), carbon);  // inlet
        add(taper(-1.1, 0.6, 0.06, 0.06, 0.42, 0.45, 0.56, 0.59, s * 0.62), accent);
    }
    // engine cover, airbox and fin
    add(taper(-1.75, -0.3, 0.3, 0.7, y0 + 0.08, 0.5, y0 + 0.08, 0.95), paint);
    add(taper(-0.3, 0.05, 0.3, 0.26, 0.7, 0.98, 0.7, 0.96), paint);
    add(taper(-1.7, -0.3, 0.025, 0.025, 0.5, 0.82, 0.92, 0.98), accent);
    add(taper(0.02, 0.06, 0.22, 0.2, 0.8, 0.95, 0.8, 0.93), carbon);  // airbox mouth
    // cockpit, driver, halo
    const helmet = add(new THREE.SphereGeometry(0.15, 18, 12), helmetMat);
    helmet.position.set(0.45, 0.78, 0);
    const visor = add(new THREE.SphereGeometry(0.152, 18, 8, -0.9, 1.8, 1.1, 0.5), carbon);
    visor.position.copy(helmet.position);
    visor.rotation.y = -Math.PI / 2;
    const halo = add(new THREE.TorusGeometry(0.36, 0.03, 8, 24, Math.PI), carbon);
    halo.rotation.set(-Math.PI / 2, 0, -Math.PI / 2);
    halo.position.set(0.55, 0.88, 0);
    halo.scale.set(1.25, 1, 1);
    add(taper(0.98, 1.05, 0.05, 0.05, 0.62, 0.66, 0.62, 0.88), carbon).position.x = 0;
    // front wing: main plane, flap, endplates, pylons
    add(taper(2.95, 3.45, 1.95, 1.95, 0.08, 0.11, 0.07, 0.1), carbon);
    add(taper(2.85, 3.1, 1.9, 1.9, 0.14, 0.17, 0.11, 0.14), accent);
    for (const s of [-1, 1]) add(box(2.82, 3.48, 0.06, 0.3, s * 0.975 - 0.012, s * 0.975 + 0.012), paint);
    for (const s of [-1, 1]) add(box(2.7, 2.95, 0.12, 0.3, s * 0.1 - 0.015, s * 0.1 + 0.015), carbon);
    // rear wing: main plane, flap, endplates, pylon
    add(taper(-2.3, -1.98, 1.0, 1.0, 0.82, 0.86, 0.84, 0.89), carbon);
    add(taper(-2.2, -2.0, 1.0, 1.0, 0.92, 0.95, 0.98, 1.0), accent);
    for (const s of [-1, 1]) add(box(-2.35, -1.9, 0.45, 1.03, s * 0.505 - 0.015, s * 0.505 + 0.015), paint);
    add(box(-2.15, -1.95, 0.3, 0.84, -0.03, 0.03), carbon);
    // suspension arms
    const arm = (x, zIn, zOut, y) => {
        const len = zOut - zIn;
        const m = add(new THREE.CylinderGeometry(0.018, 0.018, len, 6), carbon);
        m.rotation.x = Math.PI / 2;
        m.position.set(x, y, zIn + len / 2);
    };
    for (const s of [-1, 1]) {
        arm(LF, s * 0.28, s * 0.72, 0.32); arm(LF - 0.1, s * 0.28, s * 0.72, 0.42);
        arm(-LR, s * 0.35, s * 0.68, 0.3); arm(-LR + 0.1, s * 0.35, s * 0.68, 0.42);
    }

    // wheels: front ones steer
    const wheels = [];
    const addWheel = (x, z, w, steer) => {
        const pivot = new THREE.Group();
        pivot.position.set(x, R_WHEEL, z);
        const wh = wheel(w, tyre, rim, stripe);
        wh.traverse(o => { if (o.isMesh) o.castShadow = true; });
        pivot.add(wh);
        car.add(pivot);
        wheels.push({ pivot, wheel: wh, steer });
    };
    addWheel(LF, -0.8, 0.3, true); addWheel(LF, 0.8, 0.3, true);
    addWheel(-LR, -0.78, 0.4, false); addWheel(-LR, 0.78, 0.4, false);

    let spin = 0;
    car.userData.update = (steerRad, speed, dt) => {
        spin -= speed * dt / R_WHEEL;
        for (const w of wheels) {
            if (w.steer) w.pivot.rotation.y = steerRad;
            w.wheel.rotation.z = spin;
        }
    };
    car.userData.materials = { paint, accent, helmet: helmetMat };
    return car;
}
