// The environments the agent can be put in. Each changes the physics, never what
// the agent can sense, so the pretrained networks still apply and have to adapt.
// Normal friction 0.6 is the G1 feet's own (they have contact priority).
// Settings chosen from probes where the frozen agent falls within a few seconds
// and the learning agent recovers in a few hundred thousand steps.
export const WORLDS = {
    normal:   { label: 'Normal',      hud: 'Normal floor',     friction: 0.6,  leg: 1.0, pack: 0 },
    ice:      { label: 'Ice',         hud: 'Ice · μ 0.15',     friction: 0.15, leg: 1.0, pack: 0 },
    hurt:     { label: 'Injured leg', hud: 'Right leg at 20%', friction: 0.6,  leg: 0.2, pack: 0 },
    backpack: { label: 'Backpack',    hud: 'Backpack · 15 kg', friction: 0.6,  leg: 1.0, pack: 15 }
};

export function applyWorld(env, name) {
    const w = WORLDS[name];
    env.setFriction(w.friction);
    env.setLegStrength('right', w.leg);
    env.setBackpack(w.pack);
    env.world = name;
}
