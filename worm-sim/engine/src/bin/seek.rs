//! Food-seeking trials on agar with the behavior layer (behavior.rs).
//! usage: seek <model dir> <seconds> <food distance> <food angle deg> <seed> <out.csv>
//! The worm starts straight at the origin, head toward -z; the food is placed at the
//! given distance and angle from the head direction (0 = straight ahead, + = dorsal side).
//! Brain and body share one clock (24 body steps per control step); the brain plays
//! BAAIWorm's recorded input, so its muscles are the same in every trial and are cached
//! in BRAIN_CACHE=<file> (default target/brain_cache.bin).
//! env: AGAR=c_t,c_n (default 0.003,0.09); P_<NAME>=value sets a behavior parameter
//! SKIP=<s> starts from the brain's state this long after v = -65 mV (default 3).
//! SYNTH=f,λ,amp drives the body with an ideal traveling wave (Hz, body lengths, activation).
//! BALANCE=0 keeps SoftCon's uneven muscle quadrants (see Body::balance_muscles); TRIM=0 keeps the gait's steady head bend (see behavior.rs). (P_K_VANE, P_TURN_RATE, P_K_TURN, P_SIGMA, ...).
use std::io::Write;
use worm_engine::{
    behavior::{gait_trim, Mode, Params, Seek},
    body::Body,
    brain::Brain,
    worm::Worm,
};

fn f64s(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn env(name: &str, v: f64) -> f64 {
    std::env::var(name).map(|s| s.parse().unwrap()).unwrap_or(v)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dir = &a[1];
    let secs: f64 = a[2].parse().unwrap();
    let dist: f64 = a[3].parse().unwrap();
    let ang: f64 = a[4].parse::<f64>().unwrap().to_radians();
    let seed: u64 = a[5].parse().unwrap();
    let names: Vec<String> = {
        let s = std::fs::read_to_string(format!("{dir}/brain.json")).unwrap();
        let (i, j) = (s.find('[').unwrap(), s.find(']').unwrap());
        s[i + 1..j].split(',').map(|x| x.trim().trim_matches('"').to_string()).collect()
    };
    let brain = Brain::new(&f64s(&format!("{dir}/brain.bin")), 5.0 / 3.0);
    let body = Body::new(&f64s(&format!("{dir}/body.bin")));
    let wout = f64s(&format!("{dir}/wout_offline.bin"));
    let mut w = Worm::new(brain, body, wout, &names);
    w.body_steps = 24;
    let dt = w.body_steps as f64 / 240.0;
    let steps = (secs / dt) as usize;
    let ag: Vec<f64> = std::env::var("AGAR").unwrap_or("0.003,0.09".into()).split(',').map(|x| x.parse().unwrap()).collect();
    w.body.set_agar(Some((ag[0], ag[1])));
    if env("BALANCE", 1.0) != 0.0 {
        w.body.balance_muscles();
    }

    // the brain's muscles, the same in every trial
    let cache = std::env::var("BRAIN_CACHE").unwrap_or("target/brain_cache.bin".into());
    let mut muscles = std::fs::read(&cache).map(|b| b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f64>>()).unwrap_or_default();
    // the brain's first seconds from v = -65 mV swing the head round by 130°, which sends
    // trials off in different directions at random; start from its steady gait, fading in
    let skip = env("SKIP", 3.0) / dt;
    let skip = skip as usize;
    let fade = (1.0 / dt) as usize;
    let trim_steps = (60.0 / dt) as usize; // the gait's mean bend, over its first minute
    let need = steps.max(trim_steps) + skip;
    if muscles.len() < need * 96 {
        eprintln!("computing {steps} brain steps into {cache}");
        w.brain.init(-65.0);
        muscles.clear();
        for _ in 0..need {
            w.think();
            muscles.extend_from_slice(&w.muscle);
        }
        std::fs::write(&cache, muscles.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }

    let mut p = Params::default();
    let set = |n: &str, v: &mut f64| *v = env(&format!("P_{n}"), *v);
    set("SIGMA", &mut p.sigma);
    set("K_VANE", &mut p.k_vane);
    set("TAU_VANE", &mut p.tau_vane);
    set("TURN_RATE", &mut p.turn_rate);
    set("K_TURN", &mut p.k_turn);
    set("TAU_TURN", &mut p.tau_turn);
    set("REVERSE_TIME", &mut p.reverse_time);
    set("OMEGA_TIME", &mut p.omega_time);
    set("OMEGA_BIAS", &mut p.omega_bias);
    set("REFRACTORY", &mut p.refractory);
    set("STRAIGHTEN", &mut p.straighten);
    p.cone = env("P_CONE", 0.0) != 0.0;
    p.vane_segments = env("P_VANE_SEGMENTS", p.vane_segments as f64) as usize;
    p.omega_segments = env("P_OMEGA_SEGMENTS", p.omega_segments as f64) as usize;
    // head direction at rest is -z; + angle toward dorsal (+x)
    let food = [dist * ang.sin(), 0.0, -dist * ang.cos()];
    let mut seek = Seek::new(food, p, seed);
    if env("TRIM", 1.0) != 0.0 {
        seek.trim = gait_trim(&muscles[skip * 96..(skip + trim_steps) * 96], 0);
    }
    w.seek = Some(seek);

    let mirror = env("MIRROR", 0.0) != 0.0;
    let synth = std::env::var("SYNTH").ok().map(|v| {
        let v: Vec<f64> = v.split(',').map(|x| x.parse().unwrap()).collect();
        (v[0], v[1], v[2])
    });
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[6]).unwrap());
    writeln!(out, "t,mode,head_x,head_z,com_x,com_y,com_z,dlogc,bias,head_angle,turns").unwrap();
    for k in 0..steps {
        let mut b = [0.0; 96];
        b.copy_from_slice(&muscles[(k + skip) * 96..(k + skip) * 96 + 96]);
        if k < fade {
            b.iter_mut().for_each(|x| *x *= k as f64 / fade as f64);
        }
        if let Some((f, lam, amp)) = synth {
            // an ideal traveling wave instead of the brain: dorsal and ventral in antiphase
            let t = k as f64 * dt;
            for s in 0..24 {
                let ph = 2.0 * std::f64::consts::PI * (f * t - s as f64 / 24.0 / lam);
                let d = amp * ph.sin();
                for (q, sign) in [(0, 1.0), (1, -1.0), (2, 1.0), (3, -1.0)] {
                    b[q * 24 + s] = (amp + sign * d).clamp(0.0, 0.8);
                }
            }
        }
        if mirror {
            // dorsal <-> ventral (DR, VR, DL, VL)
            for s in 0..24 {
                b.swap(s, 24 + s);
                b.swap(48 + s, 72 + s);
            }
        }
        w.muscle.copy_from_slice(&b);
        w.moving();
        let seek = w.seek.as_ref().unwrap();
        let h = w.head();
        let mut c = [0.0; 3];
        for i in 0..w.body.nv {
            for j in 0..3 {
                c[j] += w.body.x[i * 3 + j] / w.body.nv as f64;
            }
        }
        let mode = match seek.mode {
            Mode::Forward => 0,
            Mode::Reverse => 1,
            Mode::Omega => 2,
        };
        writeln!(out, "{:.2},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.4},{:.4},{}", (k + 1) as f64 * dt, mode, h[0], h[2], c[0], c[1], c[2], seek.dlogc, seek.bias, seek.head_angle, seek.turns).unwrap();
    }
}
