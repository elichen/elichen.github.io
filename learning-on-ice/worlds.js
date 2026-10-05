// The environments the agent can be put in. Each changes the physics, never what
// the agent can sense, so the pretrained networks still apply and have to adapt.
// Settings chosen from probes where the frozen agent falls within about a
// second and the learning agent recovers in a few hundred thousand steps.
export const WORLDS = {
    normal:   { label: 'Normal',      hud: 'Normal floor',     friction: 1.0,  leg: 1.0, pack: 0 },
    ice:      { label: 'Ice',         hud: 'Ice · μ 0.02',     friction: 0.02, leg: 1.0, pack: 0 },
    hurt:     { label: 'Injured leg', hud: 'Right leg at 50%', friction: 1.0,  leg: 0.5, pack: 0 },
    backpack: { label: 'Backpack',    hud: 'Backpack · 10 kg', friction: 1.0,  leg: 1.0, pack: 10 }
};

export function applyWorld(env, name) {
    const w = WORLDS[name];
    env.setFriction(w.friction);
    env.setLegStrength('right', w.leg);
    env.setBackpack(w.pack);
    env.world = name;
}
