// What stands on the kitchen table: one list for the picture (scene.js) and for the physics (worker.js and the tools add it to the body's model).
// Sizes in cm, z up; the table top is z = 0. The fly collides with all of it, and with the swatter. Shapes for the physics are simple and convex:
// the bowl and the plate are filled (the fruit and the crumbs are in them), the mug has a lid of air.
const TURN = 1.25, C = Math.cos(TURN), S = Math.sin(TURN);
export const THINGS = {
    bowl: { x: 2, y: -3, wall: [[5, 0], [7, 0.5], [11, 3.2], [13, 6.5]] },            // the outer wall: radius, height
    apples: [[-1.5, -5, 4.3, 3.6], [5.5, -4.5, 4.3, 3.5], [2, 0, 4.4, 3.6]],        // x, y, z, radius
    mug: { x: -33, y: 16, r: 4, h: 8.8, handle: { out: 4.6, r: 2, thick: 0.42, z: 4.5 } },
    plate: { x: 28, y: 13, wall: [[6, 0], [10, 1], [11, 1.5], [10.8, 1.65]] },
    knife: { x: 34, y: -7, z: 0.55, turn: TURN, blade: [11, 1.5, 0.08], hilt: [0.55, 8] },   // blade: length, width, thickness; hilt: radius, length
    jar: { x: 36, y: -19, r: 3.6, h: 10.2 },
};
export const SWATTER = { half: 3, thick: 0.1 };   // cm: half-width of the square, and its thickness
// a cylinder round each thing: x, y, radius, height. High in the air the physics leaves contacts out, except inside one of these (life.js).
export const BOUNDS = [[2, -3, 13, 8], [-33, 16, 7.1, 8.8], [28, 13, 11, 1.65], [34, -7, 11, 1.1], [36, -19, 3.6, 10.2]];
export const nearThing = (x, y, z, margin = 1) => BOUNDS.some(([a, b, r, h]) => z < h + margin && Math.hypot(x - a, y - b) < r + margin);

// the body's MuJoCo XML with the things in it. Group 3: the page draws its own picture of them.
export function addThings(xml) {
    const T = THINGS, n = x => +x.toFixed(4), geoms = [], meshes = [];
    const solid = 'contype="1" conaffinity="1" group="3" friction="0.5" solref="0.001 1" solimp="0.95 0.99 0.01"';
    const hull = (name, { x, y, wall }) => {     // rings of 24 points; MuJoCo collides with their convex hull
        const v = [];
        for (const [r, z] of wall) for (let i = 0; i < 24; i++) v.push(n(r * Math.cos(i * Math.PI / 12)), n(r * Math.sin(i * Math.PI / 12)), z);
        meshes.push(`<mesh name="${name}" vertex="${v.join(' ')}" />`);
        geoms.push(`<geom name="${name}" type="mesh" mesh="${name}" pos="${x} ${y} 0" ${solid} />`);
    };
    hull('thing_bowl', T.bowl); hull('thing_plate', T.plate);
    T.apples.forEach(([x, y, z, r], i) => geoms.push(`<geom name="thing_apple${i}" type="sphere" size="${r}" pos="${x} ${y} ${z}" ${solid} />`));
    for (const k of ['mug', 'jar']) geoms.push(`<geom name="thing_${k}" type="cylinder" size="${T[k].r} ${T[k].h / 2}" pos="${T[k].x} ${T[k].y} ${T[k].h / 2}" ${solid} />`);
    const h = T.mug.handle, hx = T.mug.x + h.out;
    [[0, -h.r, h.r, -h.r], [h.r, -h.r, h.r, h.r], [0, h.r, h.r, h.r]].forEach(([x0, z0, x1, z1], i) =>   // the handle as three bars
        geoms.push(`<geom name="thing_handle${i}" type="capsule" size="${h.thick}" fromto="${hx + x0} ${T.mug.y} ${h.z + z0} ${hx + x1} ${T.mug.y} ${h.z + z1}" ${solid} />`));
    const K = T.knife, [bl, bw, bt] = K.blade, [hr, hl] = K.hilt;
    geoms.push(`<geom name="thing_blade" type="box" size="${bl / 2} ${bw / 2} ${bt / 2}" pos="${n(K.x + bl / 2 * C)} ${n(K.y + bl / 2 * S)} ${K.z}" quat="${n(Math.cos(TURN / 2))} 0 0 ${n(Math.sin(TURN / 2))}" ${solid} />`);
    geoms.push(`<geom name="thing_hilt" type="capsule" size="${hr}" fromto="${n(K.x - hl * C)} ${n(K.y - hl * S)} ${K.z} ${K.x} ${K.y} ${K.z}" ${solid} />`);
    // the swatter: a thin square that day.js moves (a mocap body, parked under the table), and a touch sensor that reports the force on it
    const R = SWATTER.half, sw = `<body name="swatter" mocap="true" pos="0 0 -60"><geom name="swatter" type="box" size="${R} ${R} ${SWATTER.thick / 2}" ${solid} /><site name="swatter" type="box" size="${R} ${R} ${SWATTER.thick}" group="3" /></body>`;
    return xml.replace('</asset>', meshes.join('') + '</asset>').replace('</worldbody>', geoms.join('') + sw + '</worldbody>').replace('</sensor>', '<touch name="swatter" site="swatter" /></sensor>');
}
