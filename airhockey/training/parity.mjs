// Records browser physics transitions for parity.py: node parity.mjs > parity.json
import fs from 'fs';
let seed = 12345;
Math.random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const src = fs.readFileSync(new URL('../environment.js', import.meta.url), 'utf8');
const game = fs.readFileSync(new URL('../game.js', import.meta.url), 'utf8');
const move = game.slice(game.indexOf('function moveAgentPaddle'), game.indexOf('function policyAction'));
const canvas = { getContext: () => ({}) };
const Env = new Function(src + '\nreturn AirHockeyEnvironment;')();
const env = new Env(canvas);
env.reset();
const moveAgentPaddle = new Function('env', move + '\nreturn moveAgentPaddle;')(env);
const snap = () => ({
    pp: [[env.playerPaddle.x, env.playerPaddle.y], [env.aiPaddle.x, env.aiPaddle.y]],
    pv: [[env.playerPaddle.dx || 0, env.playerPaddle.dy || 0], [env.aiPaddle.dx || 0, env.aiPaddle.dy || 0]],
    kp: [env.puck.x, env.puck.y], kv: [env.puck.dx, env.puck.dy],
    t: env.state.roundFrames
});
const rows = [], mode = [0, 0], held = [[0, 0], [0, 0]];
for (let f = 0; f < 60000; f++) {
    const before = snap(), acts = [];
    for (const i of [0, 1]) {
        const p = i ? env.aiPaddle : env.playerPaddle;
        if (Math.random() < .03) mode[i] = Math.floor(Math.random() * 3);
        if (Math.random() < .1) held[i] = [Math.random() * 2 - 1, Math.random() * 2 - 1];
        const sy = i ? -1 : 1;
        const chase = [Math.max(-1, Math.min(1, (env.puck.x - p.x) / 10)), Math.max(-1, Math.min(1, (env.puck.y - p.y) / 10)) * sy];
        acts.push(f % 15000 > 13500 ? [0, 0] : mode[i] === 0 ? chase : mode[i] === 1 ? held[i] : [0, 0]);
    }
    moveAgentPaddle(env.aiPaddle, acts[1], true);
    moveAgentPaddle(env.playerPaddle, acts[0], false);
    const r = env.update();
    rows.push({ s: before, a: acts, n: snap(), goal: r === 'top' ? 1 : r === 'bottom' ? -1 : 0, tout: r === 'timeout' });
}
process.stdout.write(JSON.stringify(rows));
