//! Evolution strategy over the anatomical circuit's free parameters (circuit.rs), so the
//! circuit itself makes the crawl: NMJ and proprioceptive gains by class, release curve,
//! tonic input, and gains on the commonest groups of synapses onto motor neurons.
//! Antithetic sampling, centered ranks, Adam (as grand-prix/training/es.py).
//!
//! usage: es <model dir> <out dir> [generations]
//! env: PAIRS (24), SIGMA (0.1), LR (0.03), SECS (12), SKIP (2), THREADS, GROUPS (40),
//!      (GROUPS is a special variable in bash: `GROUPS=0 cmd` is silently dropped, use `env GROUPS=0 cmd`;
//!      the first line of output gives the parameter count)
//!      CYCLE=1 plays the recorded input cycle (the rhythm BAAIWorm's gait comes from),
//!      INIT=<params.txt> to start from a saved center, SEED,
//!      INSPECT=<prefix> runs the center once and dumps motor / muscle / center per step,
//!      held-out tests: ABLATE=DB,VB (kill classes), AGAR=<drag scale>, WATER=1 (SoftCon water)
//!      SHUFFLE=rewire|sign (+ SHUFFLE_SEED): control circuit with the wiring shuffled (Brain::shuffle)
//!      V2=1: muscle time constant evolves too, and frequency / wavelength count only as far as
//!      the body bends (without it, a barely bending body could match them for free)
//!      V3=1 (implies V2): the mid and rear body (segments 8-21) must drive the bending itself:
//!      every V2 winner on the real wiring silenced those muscles and crawled with the head alone
//! fitness = score(): the gait against crawling on agar (TARGET), not raw speed
//! writes <out>/log.csv (gen, center score, mean, best, center v f λ amp len dv body alt) and <out>/gen_NNNN.txt (center)
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use worm_engine::{
    body::Body,
    brain::Brain,
    circuit::{class_of, Circuit, Params, CLASSES, REST},
    frame::fit,
    worm::{Worm, OUTPUT},
};

fn f64s(path: &str) -> Vec<f64> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn env(name: &str, v: f64) -> f64 {
    std::env::var(name).map(|s| s.parse().unwrap()).unwrap_or(v)
}

struct Model {
    brain: Vec<f64>,
    body: Vec<f64>,
    nmj: Vec<f64>,
    names: Vec<String>,
    /// synapse groups: (kind 0 gap / 1 exc / 2 inh, pre class, post class) -> member indices
    groups: Vec<((u8, String, String), Vec<usize>)>,
}

// nmj gains, v_half, v_slope / bias, prop gains, tonic (+ muscle time constant with V2)
fn n_fixed() -> usize {
    12 + 12 + 2 + 12 + 1 + v2() as usize
}

fn v2() -> bool {
    env("V2", 0.0) != 0.0 || v3()
}

fn v3() -> bool {
    env("V3", 0.0) != 0.0
}

/// unit-scaled parameter vector -> circuit params and per-group gains
fn decode(x: &[f64]) -> (Params, Vec<f64>) {
    let mut p = Params::default();
    for k in 0..12 {
        p.nmj_gain[k] = 0.3 * x[k].exp();
        p.v_half[k] = REST[k] + 5.0 * x[12 + k];
        p.prop_gain[k] = 0.1 * x[26 + k];
    }
    p.v_slope = 3.0 * (0.5 * x[24]).exp();
    p.bias = 0.1 * x[25];
    p.tonic = -0.04 + 0.03 * x[38];
    if v2() {
        p.tau_muscle = 0.05 * x[39].exp();
    }
    p.cycle = env("CYCLE", 0.0) != 0.0;
    (p, x[n_fixed()..].iter().map(|t| t.exp()).collect())
}

fn shuffle(b: &mut Brain) {
    if let Ok(k) = std::env::var("SHUFFLE") {
        b.shuffle(&k, env("SHUFFLE_SEED", 1.0) as u64);
    }
}

fn build(m: &Model, x: &[f64]) -> Worm {
    let (p, gains) = decode(x);
    let mut brain = Brain::new(&m.brain, 5.0 / 3.0);
    shuffle(&mut brain);
    let body = Body::new(&m.body);
    let mut w = Worm::new(brain, body, vec![0.0; 80 * 96], &m.names);
    w.body_steps = 24;
    let a = env("AGAR", 1.0);
    w.body.set_agar(if env("WATER", 0.0) != 0.0 { None } else { Some((0.003 * a, 0.09 * a)) });
    w.body.balance_muscles();
    for (((kind, _, _), members), &g) in m.groups.iter().zip(&gains) {
        for &k in members {
            if *kind == 0 {
                w.brain.gj_gain[k] = g;
            } else {
                w.brain.sy_gain[k] = g;
            }
        }
    }
    w.circuit = Some(Circuit::new(&m.nmj, &OUTPUT, p));
    if let Ok(s) = std::env::var("ABLATE") {
        ablate(&mut w, m, &s.split(',').collect::<Vec<_>>());
    }
    w
}

/// laser ablation of whole classes (held-out tests): the cells lose every synapse, gap
/// junction and NMJ, and sense no bending
fn ablate(w: &mut Worm, m: &Model, classes: &[&str]) {
    let dead: Vec<bool> = m.names.iter().map(|n| classes.contains(&class_of(n).as_str())).collect();
    let gone = |b: &Brain, node: u32| dead[b.cell_of[node as usize] as usize];
    for k in 0..w.brain.n_syn() {
        let (a, c, _) = w.brain.syn_pre_post(k);
        if gone(&w.brain, a) || gone(&w.brain, c) {
            w.brain.sy_gain[k] = 0.0;
        }
    }
    for k in 0..w.brain.n_gap() {
        let (a, c) = w.brain.gap_pre_post(k);
        if gone(&w.brain, a) || gone(&w.brain, c) {
            w.brain.gj_gain[k] = 0.0;
        }
    }
    let c = w.circuit.as_mut().unwrap();
    for (j, n) in OUTPUT.iter().enumerate() {
        if classes.contains(&class_of(n).as_str()) {
            c.ablate(j);
        }
    }
}

/// gait of one run: forward speed (body lengths/s, 0.63 units = 1 L), undulation frequency
/// (Hz), wavelength and curvature amplitude (centerline lengths), length / rest length, and
/// ln(dorsal / ventral) swing of the body muscles (0 when both sides work alike)
#[derive(Clone, Copy, Default)]
struct Gait {
    v: f64,
    f: f64,
    lambda: f64,
    amp: f64,
    len: f64,
    dv: f64,
    /// swing (SD over time) of the dorsal and of the ventral muscles, mean over segments 8-21
    body: f64,
    /// correlation of dorsal with ventral activation there, mean over segments (-1 = alternating)
    alt: f64,
}

/// crawling on agar, roughly: 0.5 Hz, wavelength 0.65 L, curvature amplitude ~7 /L
/// (Karbowski et al. 2006, Fang-Yen et al. 2010, Pierce-Shimomura et al. 2008), ~0.15 L/s
const TARGET: Gait = Gait { v: 0.15, f: 0.5, lambda: 0.65, amp: 7.0, len: 1.0, dv: 0.0, body: 0.1, alt: -1.0 };

/// 1 at the target gait; speed counts only up to the target, the rest by log ratio, and
/// shortening by more than 3% is penalized (both sides contracting at once), and so is one
/// side doing the bending alone (dorsal and ventral muscles alternate in the worm)
fn score(g: &Gait) -> f64 {
    let t = TARGET;
    let lr = |a: f64, b: f64| if a > 0.0 && a.is_finite() { (a / b).ln().abs() } else { 3.0 };
    // V2: below half the target amplitude, frequency and wavelength fade toward the invalid 3
    let bent = if v2() { (g.amp / (0.5 * t.amp)).clamp(0.0, 1.0) } else { 1.0 };
    let shape = |a: f64, b: f64| bent * lr(a, b) + (1.0 - bent) * 3.0;
    g.v.min(t.v) / t.v - shape(g.f, t.f) - shape(g.lambda, t.lambda) - 0.5 * lr(g.amp, t.amp) - 10.0 * (0.97 - g.len).max(0.0) - g.dv.abs().min(3.0)
        - if v3() { (t.body / g.body.max(1e-9)).ln().clamp(0.0, 3.0) + (1.0 + g.alt) / 2.0 } else { 0.0 }
}

/// signed curvature (1/L, L = this centerline's length) at slices 1..n-1, in the agar plane
/// (x-z), after Gaussian smoothing along the body as tools/kinematics.py does (σ 1.5 of its
/// 50 samples = 1.2 slices)
fn curvature(c: &[[f64; 3]]) -> (Vec<f64>, f64) {
    let len: f64 = c.windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][2] - w[0][2])).sum();
    let n = c.len() as i64;
    let sm: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            let (mut p, mut wsum) = ([0.0; 2], 0.0);
            for j in -4..=4i64 {
                let w = (-0.5 * (j as f64 / 1.2).powi(2)).exp();
                let q = c[(i + j).clamp(0, n - 1) as usize];
                p = [p[0] + w * q[0], p[1] + w * q[2]];
                wsum += w;
            }
            [p[0] / wsum, p[1] / wsum]
        })
        .collect();
    let d: Vec<[f64; 2]> = sm.windows(2).map(|w| [w[1][0] - w[0][0], w[1][1] - w[0][1]]).collect();
    let k = d
        .windows(2)
        .map(|w| {
            let (a, b) = (w[0], w[1]);
            let ds = (a[0].hypot(a[1]) + b[0].hypot(b[1])) / 2.0;
            (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1]) / ds * len
        })
        .collect();
    (k, len)
}

/// frequency = peak of the curvature power summed over the mid-body; wavelength from the
/// slope of that component's phase along the body
fn gait_shape(kappa: &[Vec<f64>], fps: f64) -> (f64, f64, f64) {
    let (n, ns) = (kappa.len(), kappa[0].len());
    let mid: Vec<usize> = (ns / 5..ns - ns / 5).collect(); // 20-80% of the body, as kinematics.py
    let mean: Vec<f64> = (0..ns).map(|i| kappa.iter().map(|k| k[i]).sum::<f64>() / n as f64).collect();
    let comp = |f: f64, i: usize| {
        let (mut re, mut im) = (0.0, 0.0);
        for (t, k) in kappa.iter().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * f * t as f64 / fps;
            re += (k[i] - mean[i]) * ph.cos();
            im -= (k[i] - mean[i]) * ph.sin();
        }
        (re, im)
    };
    let mut best = (0.0, 0.0);
    for j in 0..=98 {
        let f = 0.1 + 0.05 * j as f64; // 0.1-5 Hz
        let p: f64 = mid.iter().map(|&i| {
            let (a, b) = comp(f, i);
            a * a + b * b
        }).sum();
        if p > best.1 {
            best = (f, p);
        }
    }
    let f = best.0;
    // unwrapped phase along the body, least-squares slope per body length
    let mut ph: Vec<f64> = vec![];
    for &i in &mid {
        let (a, b) = comp(f, i);
        let mut p = b.atan2(a);
        if let Some(&q) = ph.last() {
            while p - q > std::f64::consts::PI { p -= 2.0 * std::f64::consts::PI; }
            while p - q < -std::f64::consts::PI { p += 2.0 * std::f64::consts::PI; }
        }
        ph.push(p);
    }
    let s: Vec<f64> = mid.iter().map(|&i| (i as f64 + 1.0) / (ns as f64 + 1.0)).collect();
    let (sm, pm) = (s.iter().sum::<f64>() / s.len() as f64, ph.iter().sum::<f64>() / ph.len() as f64);
    let slope = s.iter().zip(&ph).map(|(a, b)| (a - sm) * (b - pm)).sum::<f64>() / s.iter().map(|a| (a - sm) * (a - sm)).sum::<f64>();
    let rms = (mid.iter().map(|&i| kappa.iter().map(|k| (k[i] - mean[i]).powi(2)).sum::<f64>() / n as f64).sum::<f64>() / mid.len() as f64).sqrt();
    (f, 2.0 * std::f64::consts::PI / slope.abs(), std::f64::consts::SQRT_2 * rms)
}

fn evaluate(m: &Model, x: &[f64], secs: f64, skip: f64, dump: Option<&str>) -> Gait {
    let mut w = build(m, x);
    let mut files = dump.map(|p| ["motor", "muscle", "center"].map(|k| std::fs::File::create(format!("{p}_{k}.bin")).unwrap()));
    for _ in 0..20 {
        w.circuit_step(false); // 2 s of brain on a held body: the start from -65 mV
    }
    let cps = 240.0 / w.body_steps as f64;
    let len = 0.63 * 1.0; // body units per body length (1 mm)
    let rest_len = curvature(&w.body.centerline()).1;
    let com = |w: &Worm| {
        let x = &w.body.x;
        let n = x.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..n {
            for k in 0..3 {
                c[k] += x[i * 3 + k] / n as f64;
            }
        }
        c
    };
    let mut prev = com(&w);
    let (mut fwd_sum, mut len_sum) = (0.0, 0.0);
    let mut kappa = vec![];
    let (mut ms, mut ms2, mut nm) = ([0.0; 96], [0.0; 96], 0.0);
    // per segment: sums of dorsal (mean of DR, DL) and ventral (VR, VL) activation, squares, product
    let mut seg = [[0.0; 5]; 24];
    let steps = (secs * cps) as usize;
    for k in 0..steps {
        w.circuit_step(true);
        let c = w.body.centerline();
        if let Some(f) = files.as_mut() {
            let bytes = |v: &[f64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            f[0].write_all(&bytes(&w.motor_v)).unwrap();
            f[1].write_all(&bytes(&w.muscle)).unwrap();
            f[2].write_all(&bytes(&c.concat())).unwrap();
        }
        if (k + 1) as f64 / cps > skip {
            let (kc, l) = curvature(&c);
            kappa.push(kc);
            len_sum += l;
            for (i, a) in w.muscle.iter().enumerate() {
                ms[i] += a;
                ms2[i] += a * a;
            }
            for s in 0..24 {
                let (d, v) = ((w.muscle[s] + w.muscle[48 + s]) / 2.0, (w.muscle[24 + s] + w.muscle[72 + s]) / 2.0);
                for (k, a) in [d, v, d * d, v * v, d * v].into_iter().enumerate() {
                    seg[s][k] += a;
                }
            }
            nm += 1.0;
        }
        if (k + 1) % (cps as usize / 2) == 0 {
            let (r, _) = fit(&w.body.rest, &w.body.x);
            let f = [-r[0][2], -r[1][2], -r[2][2]];
            let c = com(&w);
            let d = [c[0] - prev[0], c[1] - prev[1], c[2] - prev[2]];
            prev = c;
            if (k + 1) as f64 / cps > skip {
                fwd_sum += d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
            }
            if !c.iter().all(|v| v.is_finite()) {
                return Gait { v: -1.0, ..Default::default() };
            }
        }
    }
    let (f, lambda, amp) = gait_shape(&kappa, cps);
    // swing (SD over time) of the body muscles, segments 4-21; DR, VR, DL, VL quadrants of 24
    let sd = |q: usize| (3..21).map(|s| (ms2[q * 24 + s] / nm - (ms[q * 24 + s] / nm).powi(2)).max(0.0).sqrt()).sum::<f64>();
    let dv = ((sd(0) + sd(2) + 1e-6) / (sd(1) + sd(3) + 1e-6)).ln();
    // V3: swing and dorsal-ventral correlation of segments 8-21, and ln(D/V) over them
    let (mut body, mut alt, mut sdd_sum, mut sdv_sum) = (0.0, 0.0, 0.0, 0.0);
    for q in &seg[8..22] {
        let (md, mv) = (q[0] / nm, q[1] / nm);
        let (sdd, sdv) = ((q[2] / nm - md * md).max(0.0).sqrt(), (q[3] / nm - mv * mv).max(0.0).sqrt());
        body += (sdd + sdv) / 2.0 / 14.0;
        if sdd > 1e-4 && sdv > 1e-4 {
            alt += ((q[4] / nm - md * mv) / (sdd * sdv)).clamp(-1.0, 1.0) / 14.0;
        }
        (sdd_sum, sdv_sum) = (sdd_sum + sdd, sdv_sum + sdv);
    }
    let dv = if v3() { ((sdd_sum + 1e-6) / (sdv_sum + 1e-6)).ln() } else { dv };
    Gait { v: fwd_sum / len / (secs - skip), f, lambda, amp, len: len_sum / kappa.len() as f64 / rest_len, dv, body, alt }
}

fn load(dir: &str, n_groups: usize) -> Model {
    let names: Vec<String> = {
        let s = std::fs::read_to_string(format!("{dir}/brain.json")).unwrap();
        let (i, j) = (s.find('[').unwrap(), s.find(']').unwrap());
        s[i + 1..j].split(',').map(|x| x.trim().trim_matches('"').to_string()).collect()
    };
    let brain = f64s(&format!("{dir}/brain.bin"));
    let mut b = Brain::new(&brain, 5.0 / 3.0);
    shuffle(&mut b);
    let cls: Vec<String> = names.iter().map(|n| class_of(n)).collect();
    let motor = |c: &str| CLASSES.contains(&c);
    let mut map: HashMap<(u8, String, String), Vec<usize>> = HashMap::new();
    for k in 0..b.n_gap() {
        let (a, c) = b.gap_pre_post(k);
        let (pa, pc) = (&cls[b.cell_of[a as usize] as usize], &cls[b.cell_of[c as usize] as usize]);
        if motor(pc) {
            map.entry((0, pa.clone(), pc.clone())).or_default().push(k);
        }
    }
    for k in 0..b.n_syn() {
        let (a, c, exc) = b.syn_pre_post(k);
        let (pa, pc) = (&cls[b.cell_of[a as usize] as usize], &cls[b.cell_of[c as usize] as usize]);
        if motor(pc) {
            map.entry((if exc { 1 } else { 2 }, pa.clone(), pc.clone())).or_default().push(k);
        }
    }
    let mut groups: Vec<_> = map.into_iter().collect();
    groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
    groups.truncate(n_groups);
    Model { brain, body: f64s(&format!("{dir}/body_sym.bin")), nmj: f64s(&format!("{dir}/nmj.bin")), names, groups }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (dir, out) = (&a[1], &a[2]);
    let gens: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    std::fs::create_dir_all(out).unwrap();
    let model = load(dir, env("GROUPS", 40.0) as usize);
    let dim = n_fixed() + model.groups.len();
    {
        let mut f = std::fs::File::create(format!("{out}/groups.txt")).unwrap();
        for ((k, a, b), idx) in &model.groups {
            writeln!(f, "{} {a} {b} {}", ["gap", "exc", "inh"][*k as usize], idx.len()).unwrap();
        }
    }
    let mut x = vec![0.0; dim];
    if let Ok(p) = std::env::var("INIT") {
        x = std::fs::read_to_string(p).unwrap().split_whitespace().map(|s| s.parse().unwrap()).collect();
        assert_eq!(x.len(), dim);
    }
    if let Ok(p) = std::env::var("INSPECT") {
        // one run of the center with motor voltages, muscles and centerline dumped per step
        let g = evaluate(&model, &x, env("SECS", 12.0), env("SKIP", 2.0), Some(&p));
        println!("score {:+.3}: {:+.3} L/s, {:.2} Hz, wavelength {:.2} L, amplitude {:.1} /L, length {:.3}, ln(D/V) {:+.2}, body swing {:.3}, D-V corr {:+.2}; dumps at {p}_{{motor,muscle,center}}.bin", score(&g), g.v, g.f, g.lambda, g.amp, g.len, g.dv, g.body, g.alt);
        return;
    }
    let (pairs, sigma, lr) = (env("PAIRS", 24.0) as usize, env("SIGMA", 0.1), env("LR", 0.03));
    let (secs, skip) = (env("SECS", 12.0), env("SKIP", 2.0));
    let threads = env("THREADS", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as f64) as usize;
    let mut rng = Rng(env("SEED", 1.0) as u64 * 0x9E3779B97F4A7C15 + 1);
    let (mut mo, mut ve) = (vec![0.0; dim], vec![0.0; dim]);
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(format!("{out}/log.csv")).unwrap();
    println!("dim {dim}, {pairs} pairs, sigma {sigma}, lr {lr}, {threads} threads");
    for g in 0..gens {
        let t0 = std::time::Instant::now();
        let eps: Vec<Vec<f64>> = (0..pairs).map(|_| (0..dim).map(|_| rng.gauss()).collect()).collect();
        // candidate 0 = center, then +eps, -eps
        let cands: Vec<Vec<f64>> = std::iter::once(x.clone())
            .chain(eps.iter().flat_map(|e| [1.0, -1.0].map(|s| x.iter().zip(e).map(|(a, b)| a + s * sigma * b).collect::<Vec<f64>>())))
            .collect();
        let next = AtomicUsize::new(0);
        let gaits: Vec<Gait> = {
            let res = std::sync::Mutex::new(vec![Gait::default(); cands.len()]);
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= cands.len() {
                            break;
                        }
                        let g = evaluate(&model, &cands[i], secs, skip, None);
                        res.lock().unwrap()[i] = g;
                    });
                }
            });
            res.into_inner().unwrap()
        };
        let fit: Vec<f64> = gaits.iter().map(score).collect();
        // centered ranks over the 2 × pairs perturbations
        let pf = &fit[1..];
        let mut order: Vec<usize> = (0..pf.len()).collect();
        order.sort_by(|&i, &j| pf[i].partial_cmp(&pf[j]).unwrap());
        let mut rank = vec![0.0; pf.len()];
        for (r, &i) in order.iter().enumerate() {
            rank[i] = r as f64 / (pf.len() - 1) as f64 - 0.5;
        }
        let mut grad = vec![0.0; dim];
        for (p, e) in eps.iter().enumerate() {
            let w = rank[2 * p] - rank[2 * p + 1];
            for d in 0..dim {
                grad[d] += w * e[d] / (pairs as f64 * sigma);
            }
        }
        let t = (g + 1) as f64;
        for d in 0..dim {
            mo[d] = 0.9 * mo[d] + 0.1 * grad[d];
            ve[d] = 0.999 * ve[d] + 0.001 * grad[d] * grad[d];
            let (mh, vh) = (mo[d] / (1.0 - 0.9f64.powf(t)), ve[d] / (1.0 - 0.999f64.powf(t)));
            x[d] += lr * mh / (vh.sqrt() + 1e-8);
        }
        let best = pf.iter().cloned().fold(f64::MIN, f64::max);
        let mean = pf.iter().sum::<f64>() / pf.len() as f64;
        let c = gaits[0];
        writeln!(log, "{g},{:.5},{:.5},{:.5},{:.4},{:.3},{:.3},{:.2},{:.4},{:.3},{:.4},{:.3}", fit[0], mean, best, c.v, c.f, c.lambda, c.amp, c.len, c.dv, c.body, c.alt).unwrap();
        let s: Vec<String> = cands[0].iter().map(|v| format!("{v:.6}")).collect();
        std::fs::write(format!("{out}/gen_{g:04}.txt"), s.join(" ")).unwrap();
        println!(
            "gen {g}: center {:+.3} ({:+.3} L/s, {:.2} Hz, λ {:.2}, amp {:.1}, len {:.3}, D/V {:+.2}), mean {:+.3}, best {:+.3} ({:.0} s)",
            fit[0], c.v, c.f, c.lambda, c.amp, c.len, c.dv, mean, best, t0.elapsed().as_secs_f64()
        );
    }
}
