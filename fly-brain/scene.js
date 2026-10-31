// The kitchen table: only pictures. The fly's physics knows the flat tabletop and nothing else, so its goals stay clear of what stands here
// and it flies above it. Sizes in cm, z up; the table top is z = 0.
import * as THREE from 'three';

const std = (color, roughness = 0.6, extra = {}) => new THREE.MeshStandardMaterial({ color, roughness, ...extra });
const place = (mesh, x, y, z = 0) => { mesh.position.set(x, y, z); mesh.castShadow = mesh.receiveShadow = true; return mesh; };
const lathe = (pts, mat) => { const g = new THREE.LatheGeometry(pts.map(([r, z]) => new THREE.Vector2(r, z)), 64); g.rotateX(Math.PI / 2); return new THREE.Mesh(g, mat); };

function wood() {
    const c = document.createElement('canvas'); c.width = 2048; c.height = 1024;
    const g = c.getContext('2d');
    for (let p = 0; p < 5; p++) {
        const y0 = p * 1024 / 5, h = 1024 / 5, tone = [0, 8, -6, 5, -3][p];
        g.fillStyle = `rgb(${176 + tone},${128 + tone},${84 + tone})`; g.fillRect(0, y0, 2048, h);
        for (let i = 0; i < 90; i++) {
            const y = y0 + Math.random() * h, a = 0.03 + Math.random() * 0.1;
            g.strokeStyle = `rgba(${70 + Math.random() * 30},${40 + Math.random() * 20},${20},${a})`; g.lineWidth = 0.6 + Math.random() * 2;
            g.beginPath(); g.moveTo(0, y);
            for (let x = 0; x <= 2048; x += 128) g.lineTo(x, y + 7 * Math.sin(x / 300 + i + p) + 3 * Math.sin(x / 90 + i));
            g.stroke();
        }
        g.fillStyle = 'rgba(40,22,10,0.55)'; g.fillRect(0, y0, 2048, 2.5);
    }
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 16;
    return t;
}

function apple(x, y, z, r, color) {
    const a = new THREE.Group(), pts = [];
    for (let i = 0; i <= 24; i++) { const u = i / 24 * Math.PI, dent = 1 - 0.28 * Math.exp(-(((u - Math.PI) / 0.45) ** 2)) - 0.14 * Math.exp(-((u / 0.35) ** 2)); pts.push([Math.max(0.001, r * Math.sin(u)), -r * Math.cos(u) * dent]); }
    a.add(lathe(pts, std(color, 0.35)));
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.1, 1.1), std('#4a3420', 0.8)); stem.rotation.x = Math.PI / 2 + 0.2; stem.position.z = r * 0.95;
    a.add(stem); a.position.set(x, y, z); a.traverse(o => { o.castShadow = o.receiveShadow = true; });
    return a;
}

export function buildKitchen(scene) {
    const room = new THREE.Group();
    scene.background = new THREE.Color('#e8e2d6');
    const top = new THREE.Mesh(new THREE.BoxGeometry(100, 64, 3), std('#ffffff', 0.55, { map: wood(), envMapIntensity: 0.5 }));
    top.position.z = -1.5; top.receiveShadow = true; room.add(top);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(800, 800), std('#a79f92', 0.95)); floor.position.z = -75; room.add(floor);
    const back = new THREE.Mesh(new THREE.PlaneGeometry(800, 300), std('#efe9dc', 0.95)); back.rotation.x = Math.PI / 2; back.position.set(0, 180, 70); room.add(back);
    const glassM = new THREE.MeshPhysicalMaterial({ color: '#f4f8f6', roughness: 0.05, transmission: 0, transparent: true, opacity: 0.28, depthWrite: false });
    // fruit bowl, centre back: fly goals stay out of everything here (day.js TABLE plus KEEP_OUT)
    const bowl = lathe([[0.01, 0], [5, 0], [7, 0.5], [11, 3.2], [13, 6.5], [12.6, 6.6], [10.6, 3.6], [6.5, 1.1], [0.01, 0.9]], std('#2f5e74', 0.25));
    room.add(place(bowl, 2, -3));
    room.add(apple(-1.5, -5, 4.3, 3.6, '#a8231f'), apple(5.5, -4.5, 4.3, 3.5, '#c2a52a'), apple(2, 0, 4.4, 3.6, '#8f2a22'));
    // mug
    const mug = lathe([[0.01, 0], [3.8, 0], [4, 0.3], [4, 8.8], [3.65, 8.8], [3.65, 0.8], [0.01, 0.8]], std('#e9e3d2', 0.3));
    const handle = new THREE.Mesh(new THREE.TorusGeometry(2, 0.42, 14, 40), std('#e9e3d2', 0.3)); handle.rotation.x = Math.PI / 2; handle.position.set(4.6, 0, 4.5); handle.castShadow = true;
    const coffee = new THREE.Mesh(new THREE.CircleGeometry(3.65, 40), std('#2a1609', 0.12)); coffee.position.z = 7.5;
    const mugG = new THREE.Group(); mugG.add(mug, handle, coffee); mugG.position.set(-33, 16, 0); room.add(mugG); mug.castShadow = true;
    // a plate with crumbs, a knife, a jam jar
    const plate = lathe([[0.01, 0], [6, 0], [10, 1], [11, 1.5], [10.8, 1.65], [9.8, 1.2], [6, 0.35], [0.01, 0.35]], std('#f5f2ea', 0.25));
    room.add(place(plate, 28, 13));
    for (let i = 0; i < 40; i++) { const c = new THREE.Mesh(new THREE.DodecahedronGeometry(0.1 + Math.random() * 0.2), std('#b98a4e', 0.9)); const a = Math.random() * 6.28, r = Math.random() * 5; c.position.set(28 + r * Math.cos(a), 13 + r * Math.sin(a), 0.42); room.add(c); }
    const knife = new THREE.Group();
    const blade = new THREE.Mesh(new THREE.BoxGeometry(11, 1.5, 0.08), std('#cfd3d6', 0.2, { metalness: 0.9 })); blade.position.x = 5.5;
    const hilt = new THREE.Mesh(new THREE.CapsuleGeometry(0.55, 8, 8, 16), std('#3a2a1e', 0.5)); hilt.rotation.z = Math.PI / 2; hilt.position.x = -4;
    blade.castShadow = hilt.castShadow = true; knife.add(blade, hilt); knife.position.set(34, -7, 0.55); knife.rotation.z = 1.25; room.add(knife);
    const jar = lathe([[0.01, 0], [3.4, 0], [3.6, 0.5], [3.6, 7], [3, 8], [3, 9], [0.01, 9]], glassM); room.add(place(jar, 36, -19));
    jar.castShadow = false;
    const jam = lathe([[0.01, 0.2], [3.3, 0.2], [3.3, 5.5], [0.01, 5.5]], std('#6d1329', 0.3)); room.add(place(jam, 36, -19));
    const lid = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, 1.2, 40), std('#c6c1b4', 0.3, { metalness: 0.6 })); lid.rotation.x = Math.PI / 2; room.add(place(lid, 36, -19, 9.6));
    // a tea towel at the corner, and an old coffee ring
    const tc = document.createElement('canvas'); tc.width = 512; tc.height = 360;
    const tg = tc.getContext('2d'); tg.fillStyle = '#e6e1d3'; tg.fillRect(0, 0, 512, 360); tg.fillStyle = 'rgba(52,84,110,0.75)';
    for (let i = 0; i < 512; i += 64) tg.fillRect(i, 0, 22, 360);
    tg.fillStyle = 'rgba(52,84,110,0.4)'; for (let i = 0; i < 360; i += 64) tg.fillRect(0, i, 512, 22);
    const tt = new THREE.CanvasTexture(tc); tt.colorSpace = THREE.SRGBColorSpace;
    const towel = new THREE.Mesh(new THREE.BoxGeometry(20, 14, 0.25), std('#ffffff', 0.98, { map: tt, envMapIntensity: 0.3 })); towel.rotation.z = 0.2; room.add(place(towel, -37, -21, 0.12));
    const ring = new THREE.Mesh(new THREE.RingGeometry(3.6, 3.95, 64), new THREE.MeshBasicMaterial({ color: '#5a3517', transparent: true, opacity: 0.32, depthWrite: false })); ring.position.set(-12, 10, 0.006); room.add(ring);
    scene.add(room);
    // window light for the whole table; FlyView's own key light carries the fly's fine shadow
    const sun = new THREE.DirectionalLight('#fff3df', 1.6); sun.position.set(-50, -45, 90); sun.castShadow = true; sun.shadow.mapSize.set(4096, 4096);
    Object.assign(sun.shadow.camera, { left: -70, right: 70, top: 55, bottom: -55, near: 5, far: 260 }); sun.shadow.bias = -0.0004;
    scene.add(sun);
    return room;
}

// a drop of sugar water: a shiny dome, shrinking as it is eaten
export function sugarDrop() {
    const g = new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2); g.rotateX(Math.PI / 2);
    const m = new THREE.Mesh(g, new THREE.MeshPhysicalMaterial({ color: '#f6e7b8', roughness: 0.04, clearcoat: 1, transparent: true, opacity: 0.55, envMapIntensity: 2 }));
    m.castShadow = true;
    return m;
}

// the swatter: a wire handle and a square of mesh
export function swatter(radius) {
    const c = document.createElement('canvas'); c.width = c.height = 128;
    const g = c.getContext('2d'); g.fillStyle = '#b42a2a'; g.fillRect(0, 0, 128, 128); g.clearRect(0, 0, 0, 0);
    g.globalCompositeOperation = 'destination-out';
    for (let x = 4; x < 128; x += 8) for (let y = 4; y < 128; y += 8) { g.beginPath(); g.arc(x, y, 2.4, 0, 7); g.fill(); }
    const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
    const group = new THREE.Group();
    const head = new THREE.Mesh(new THREE.PlaneGeometry(2 * radius, 2 * radius), new THREE.MeshStandardMaterial({ map: tex, alphaTest: 0.4, side: THREE.DoubleSide, roughness: 0.6 }));
    head.castShadow = true;
    const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 34), std('#d4c9b4', 0.4)); stick.position.set(0, -radius - 17, 0); stick.castShadow = true;
    group.add(head, stick); group.visible = false;
    return group;
}
