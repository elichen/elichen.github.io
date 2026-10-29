//! BAAIWorm's body: SoftCon's projective dynamics (Metaworm/sim/fem), 984 vertices,
//! 3,341 corotated tetrahedra and 4,558 muscle fiber constraints in 96 segments,
//! swimming under SoftCon's per-triangle quadratic drag and lift.
//!
//! Each step (1/240 s): q = x + dt v + dt² M⁻¹ f_drag, then up to 50 local/global
//! iterations of
//!     (M/dt² + L) x = M/dt² q + Σ w Aᵀ p(x)
//! where each constraint projects its deformation F (or fiber stretch F f) onto
//! a target p: the nearest rotation, the nearest volume-preserving F, or the
//! fiber shortened to (1 - activation). L = Σ w AᵀA is factored once. Like
//! World::ProjectiveDynamicsMethod, the loop stops when ‖Δx‖/3n < 5e-7 and keeps
//! the iterate from before that last solve. Velocities are damped by 0.9999.
//!
//! The constraint list comes from tools/export_body.py, which repeats
//! Worm::Initialize and Muscle::Initialize. PD's matrices act the same way on
//! x, y and z, so the solve is one scalar 984×984 Cholesky with three columns.

const DT: f64 = 1.0 / 240.0;
const MAX_ITER: usize = 50;
const EPS: f64 = 5e-7;
const DAMPING: f64 = 0.9999;
const MASS: f64 = 10.0;
const YOUNG: f64 = 2e5;
const POISSON: f64 = 0.4;
const MUSCLE_K: f64 = 5e5;

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3]; // row-major

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
fn normalized(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 {
        scale(a, 1.0 / n)
    } else {
        a
    }
}
#[inline]
fn mv(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}
#[inline]
pub fn det(m: &M3) -> f64 {
    dot(m[0], cross(m[1], m[2]))
}

#[inline]
fn pt(x: &[f64], i: u32) -> V3 {
    let i = i as usize * 3;
    [x[i], x[i + 1], x[i + 2]]
}

/// Eigen-decomposition of a symmetric 3×3 by cyclic Jacobi: returns (values, vectors as columns).
fn sym_eig(m: &M3) -> (V3, M3) {
    let mut a = *m;
    let mut v: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..12 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        let diag = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if off <= 1e-30 * diag {
            break;
        }
        for &(p, q, r) in &[(0usize, 1usize, 2usize), (0, 2, 1), (1, 2, 0)] {
            let apq = a[p][q];
            if apq == 0.0 {
                continue;
            }
            // rotation zeroing a[p][q] (Numerical Recipes' jacobi)
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            a[p][p] -= t * apq;
            a[q][q] += t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let (arp, arq) = (a[r][p], a[r][q]);
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// F = U diag(s) Vᵀ with s ≥ 0 (like Eigen's JacobiSVD). Returns (U columns, s, V columns).
pub fn svd3(f: &M3) -> ([V3; 3], V3, [V3; 3]) {
    let mut guess = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    svd3_from(f, &mut guess)
}

/// svd3 starting the eigen-solve of FᵀF from `guess` (columns = last time's V,
/// updated in place): the deformation changes little between iterations.
pub fn svd3_from(f: &M3, guess: &mut M3) -> ([V3; 3], V3, [V3; 3]) {
    let mut ftf = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ftf[i][j] = f[0][i] * f[0][j] + f[1][i] * f[1][j] + f[2][i] * f[2][j];
        }
    }
    // B = Gᵀ (FᵀF) G
    let g = *guess;
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = ftf[i][0] * g[0][j] + ftf[i][1] * g[1][j] + ftf[i][2] * g[2][j];
        }
    }
    let mut bm = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            bm[i][j] = g[0][i] * t[0][j] + g[1][i] * t[1][j] + g[2][i] * t[2][j];
        }
    }
    let (lam, vj) = sym_eig(&bm);
    let mut vm = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            vm[i][j] = g[i][0] * vj[0][j] + g[i][1] * vj[1][j] + g[i][2] * vj[2][j];
        }
    }
    *guess = vm;
    // sort descending
    let mut ord = [0usize, 1, 2];
    ord.sort_by(|&a, &b| lam[b].partial_cmp(&lam[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut v = [[0.0; 3]; 3];
    let mut s = [0.0; 3];
    for (k, &o) in ord.iter().enumerate() {
        v[k] = [vm[0][o], vm[1][o], vm[2][o]];
        s[k] = lam[o].max(0.0).sqrt();
    }
    let mut u = [[0.0; 3]; 3];
    u[0] = if s[0] > 0.0 { normalized(mv(f, v[0])) } else { [1.0, 0.0, 0.0] };
    let mut u1 = mv(f, v[1]);
    u1 = sub(u1, scale(u[0], dot(u1, u[0])));
    u[1] = if norm(u1) > 1e-12 * s[0].max(1e-300) { normalized(u1) } else { normalized(cross(u[0], if u[0][0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] })) };
    let mut u2 = mv(f, v[2]);
    u2 = sub(u2, scale(u[0], dot(u2, u[0])));
    u2 = sub(u2, scale(u[1], dot(u2, u[1])));
    u[2] = if norm(u2) > 1e-9 * s[0].max(1e-300) {
        normalized(u2)
    } else {
        // F is (nearly) rank 2: pick the sign that gives F's orientation
        let c = cross(u[0], u[1]);
        let sign = if det(f) * det(&[v[0], v[1], v[2]]) < 0.0 { -1.0 } else { 1.0 };
        scale(c, sign)
    };
    (u, s, v)
}

/// Σ_i s_i u_i v_iᵀ
fn compose(u: &[V3; 3], s: V3, v: &[V3; 3]) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for k in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += s[k] * u[k][i] * v[k][j];
            }
        }
    }
    m
}

struct Tet {
    idx: [u32; 4],
    guess: M3,
    g: [V3; 4], // F = Σ x_k g_kᵀ
    w_dev: f64,
    w_vol: f64,
}

struct Fiber {
    seg: u32,
    idx: [u32; 4],
    a: [f64; 4], // F f = Σ a_k x_k
    w: f64,
}

/// Envelope (profile) Cholesky of the permuted system matrix.
struct Profile {
    n: usize,
    first: Vec<usize>, // first nonzero column of row i
    start: Vec<usize>, // offset of row i in `l`
    l: Vec<f64>,       // row i holds columns first[i]..=i
}

impl Profile {
    fn new(n: usize, entries: &[(usize, usize, f64)]) -> Profile {
        let mut first: Vec<usize> = (0..n).collect();
        for &(i, j, _) in entries {
            let (r, c) = if i > j { (i, j) } else { (j, i) };
            first[r] = first[r].min(c);
        }
        let mut start = vec![0; n + 1];
        for i in 0..n {
            start[i + 1] = start[i] + (i - first[i] + 1);
        }
        let mut l = vec![0.0; start[n]];
        for &(i, j, v) in entries {
            if i >= j {
                l[start[i] + j - first[i]] += v;
            }
        }
        // in-place Cholesky
        for i in 0..n {
            for j in first[i]..=i {
                let lo = first[i].max(first[j]);
                let mut s = l[start[i] + j - first[i]];
                for k in lo..j {
                    s -= l[start[i] + k - first[i]] * l[start[j] + k - first[j]];
                }
                if j == i {
                    assert!(s > 0.0, "matrix not positive definite");
                    l[start[i] + i - first[i]] = s.sqrt();
                } else {
                    l[start[i] + j - first[i]] = s / l[start[j] + j - first[j]];
                }
            }
        }
        Profile { n, first, start, l }
    }

    /// Solves in place for three right-hand sides stored as b[i*3 + c].
    fn solve3(&self, b: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let (f, s) = (self.first[i], self.start[i]);
            let (mut x, mut y, mut z) = (b[i * 3], b[i * 3 + 1], b[i * 3 + 2]);
            for k in f..i {
                let lk = self.l[s + k - f];
                x -= lk * b[k * 3];
                y -= lk * b[k * 3 + 1];
                z -= lk * b[k * 3 + 2];
            }
            let d = self.l[s + i - f];
            b[i * 3] = x / d;
            b[i * 3 + 1] = y / d;
            b[i * 3 + 2] = z / d;
        }
        for i in (0..n).rev() {
            let (f, s) = (self.first[i], self.start[i]);
            let d = self.l[s + i - f];
            let (x, y, z) = (b[i * 3] / d, b[i * 3 + 1] / d, b[i * 3 + 2] / d);
            b[i * 3] = x;
            b[i * 3 + 1] = y;
            b[i * 3 + 2] = z;
            for k in f..i {
                let lk = self.l[s + k - f];
                b[k * 3] -= lk * x;
                b[k * 3 + 1] -= lk * y;
                b[k * 3 + 2] -= lk * z;
            }
        }
    }
}

pub struct Body {
    pub nv: usize,
    pub rest: Vec<f64>,
    pub x: Vec<f64>,
    pub v: Vec<f64>,
    pub f_ext: Vec<f64>,
    tets: Vec<Tet>,
    fibers: Vec<Fiber>,
    pub tris: Vec<[u32; 3]>,
    pub sampling: Vec<u32>,
    pub ends: [u32; 2],
    pub contour: [u32; 3],
    /// activation of each of the 96 muscle segments (DR, VR, DL, VL × 24, head first)
    pub act: Vec<f64>,
    perm: Vec<usize>, // vertex -> row in the factor
    chol: Profile,
    m_dt2: f64,
    scratch: Vec<f64>,
    xk: Vec<f64>,
    pub iterations: usize,
    /// multiplies the water's drag and lift (1 = SoftCon's water)
    pub fluid: f64,
    /// None = SoftCon's water; Some = crawling on agar (resistive force theory)
    agar: Option<Agar>,
    entries: Vec<(usize, usize, f64)>,
    slice: Vec<usize>, // vertex -> slice along the body, for the centerline
}

/// Viscous drag of a surface groove, per vertex: c_t along the body, c_n across it and
/// out of the plane. Implicit as two projective-dynamics terms with constant weights,
/// so the matrix stays factored: w_t |x - x0|^2 + (w_n - w_t) |x - p|^2, where p is x0
/// moved along the local body tangent by (x - x0)'s tangential part.
#[derive(Clone, Copy)]
struct Agar {
    w_t: f64,
    w_n: f64,
}

const SLICES: usize = 40;

impl Body {
    pub fn new(data: &[f64]) -> Body {
        let nv = data[0] as usize;
        let nt = data[1] as usize;
        let nm = data[2] as usize;
        let ntri = data[3] as usize;
        let ns = data[4] as usize;
        let mut o = 5;
        let rest = data[o..o + nv * 3].to_vec();
        o += nv * 3;
        let mu = YOUNG / (1.0 + POISSON);
        let grows = |inv: &[f64]| -> [V3; 4] {
            let r0 = [inv[0], inv[1], inv[2]];
            let r1 = [inv[3], inv[4], inv[5]];
            let r2 = [inv[6], inv[7], inv[8]];
            [scale(add(add(r0, r1), r2), -1.0), r0, r1, r2]
        };
        let mut tets = Vec::with_capacity(nt);
        for _ in 0..nt {
            let r = &data[o..o + 14];
            o += 14;
            let vol = r[4];
            tets.push(Tet {
                idx: [r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u32],
                guess: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                g: grows(&r[5..14]),
                w_dev: mu * vol,
                w_vol: mu * vol * POISSON,
            });
        }
        let mut fibers = Vec::with_capacity(nm);
        for _ in 0..nm {
            let r = &data[o..o + 19];
            o += 19;
            let g = grows(&r[6..15]);
            let f = [r[15], r[16], r[17]];
            fibers.push(Fiber {
                seg: r[0] as u32,
                idx: [r[1] as u32, r[2] as u32, r[3] as u32, r[4] as u32],
                a: [dot(g[0], f), dot(g[1], f), dot(g[2], f), dot(g[3], f)],
                w: r[5] * MUSCLE_K * r[18],
            });
        }
        let mut tris = Vec::with_capacity(ntri);
        for _ in 0..ntri {
            tris.push([data[o] as u32, data[o + 1] as u32, data[o + 2] as u32]);
            o += 3;
        }
        let sampling = data[o..o + ns].iter().map(|&x| x as u32).collect();
        o += ns;
        let ends = [data[o] as u32, data[o + 1] as u32];
        let contour = [data[o + 2] as u32, data[o + 3] as u32, data[o + 4] as u32];

        // order vertices along the body (z) so the factor is a narrow band
        let mut order: Vec<usize> = (0..nv).collect();
        order.sort_by(|&a, &b| rest[a * 3 + 2].partial_cmp(&rest[b * 3 + 2]).unwrap());
        let mut perm = vec![0; nv];
        for (row, &vtx) in order.iter().enumerate() {
            perm[vtx] = row;
        }
        let m_dt2 = MASS / nv as f64 / (DT * DT);
        let mut entries = Vec::new();
        for i in 0..nv {
            entries.push((i, i, m_dt2));
        }
        for t in &tets {
            let w = t.w_dev + t.w_vol;
            for a in 0..4 {
                for b in 0..4 {
                    entries.push((perm[t.idx[a] as usize], perm[t.idx[b] as usize], w * dot(t.g[a], t.g[b])));
                }
            }
        }
        for f in &fibers {
            for a in 0..4 {
                for b in 0..4 {
                    entries.push((perm[f.idx[a] as usize], perm[f.idx[b] as usize], f.w * f.a[a] * f.a[b]));
                }
            }
        }
        let chol = Profile::new(nv, &entries);
        let (z0, z1) = (0..nv).map(|i| rest[i * 3 + 2]).fold((f64::MAX, f64::MIN), |(a, b), z| (a.min(z), b.max(z)));
        let slice = (0..nv).map(|i| (((rest[i * 3 + 2] - z0) / (z1 - z0) * SLICES as f64) as usize).min(SLICES - 1)).collect();
        Body {
            nv,
            x: rest.clone(),
            v: vec![0.0; nv * 3],
            f_ext: vec![0.0; nv * 3],
            rest,
            tets,
            fibers,
            tris,
            sampling,
            ends,
            contour,
            act: vec![0.0; 96],
            perm,
            chol,
            m_dt2,
            scratch: vec![0.0; nv * 3],
            xk: vec![0.0; nv * 3],
            iterations: 0,
            fluid: 1.0,
            agar: None,
            entries,
            slice,
        }
    }

    /// Crawl on agar with per-vertex drag coefficients c_t (along the body) and c_n
    /// (across it), or swim in SoftCon's water with None.
    pub fn set_agar(&mut self, c: Option<(f64, f64)>) {
        self.agar = c.map(|(ct, cn)| Agar { w_t: ct / DT, w_n: cn / DT });
        let mut e = self.entries.clone();
        if let Some(a) = self.agar {
            for i in 0..self.nv {
                e.push((i, i, a.w_n));
            }
        }
        self.chol = Profile::new(self.nv, &e);
    }

    /// unit tangent of the centerline at each slice, from positions x
    fn tangents(&self, x: &[f64]) -> Vec<V3> {
        let mut c = vec![[0.0; 3]; SLICES];
        let mut n = vec![0.0f64; SLICES];
        for i in 0..self.nv {
            let k = self.slice[i];
            c[k] = add(c[k], pt(x, i as u32));
            n[k] += 1.0;
        }
        for k in 0..SLICES {
            c[k] = scale(c[k], 1.0 / n[k].max(1.0));
        }
        (0..SLICES).map(|k| normalized(sub(c[(k + 1).min(SLICES - 1)], c[k.saturating_sub(1)]))).collect()
    }

    pub fn reset(&mut self) {
        self.x.copy_from_slice(&self.rest);
        for v in self.v.iter_mut() {
            *v = 0.0;
        }
    }

    pub fn factor_nnz(&self) -> usize {
        self.chol.l.len()
    }


    /// Worm::ComputeDragForces
    pub fn drag(&mut self) {
        let f = &mut self.f_ext;
        for v in f.iter_mut() {
            *v = 0.0;
        }
        const CL_X: [f64; 5] = [-90.0, -10.0, -5.0, 15.0, 90.0];
        const CL_Y: [f64; 5] = [-0.5, -1.26, 0.0, 1.8, 0.5];
        const MAX_F: f64 = 1.0e3;
        let cl = |aoa: f64| -> f64 {
            let th = aoa * 180.0 / std::f64::consts::PI;
            if th.abs() > 85.0 {
                return 0.0;
            }
            for k in 0..4 {
                let last = k >= 2;
                if CL_X[k] <= th && (th < CL_X[k + 1] || (last && th <= CL_X[k + 1])) {
                    let t = (th - CL_X[k]) / (CL_X[k + 1] - CL_X[k]);
                    return (1.0 - t) * CL_Y[k] + t * CL_Y[k + 1];
                }
            }
            0.0
        };
        for tri in &self.tris {
            let (p1, p2, p3) = (pt(&self.x, tri[0]), pt(&self.x, tri[1]), pt(&self.x, tri[2]));
            let (v1, v2, v3) = (pt(&self.v, tri[0]), pt(&self.v, tri[1]), pt(&self.v, tri[2]));
            let v_rel = scale(add(add(v1, v2), v3), -1.0 / 3.0);
            let speed = norm(v_rel);
            if speed < 1e-6 {
                continue;
            }
            let vn = scale(v_rel, 1.0 / speed);
            let c = cross(sub(p2, p1), sub(p3, p1));
            let area = 0.5 * norm(c);
            let d = scale(normalized(c), -1.0);
            let ddv = dot(d, vn);
            if ddv <= 0.0 {
                continue;
            }
            let aoa = 0.5 * std::f64::consts::PI - ddv.clamp(-1.0, 1.0).acos();
            let q = 0.5 * 1000.0 * area * speed * speed * self.fluid;
            let cd = 0.5 * (1.0 - (2.0 * aoa).cos());
            let mut fd = scale(vn, q * cd);
            let d_lift = cross(normalized(cross(vn, d)), vn);
            let mut fl = scale(d_lift, q * cl(aoa));
            if norm(fd) > MAX_F {
                fd = scale(normalized(fd), MAX_F);
            }
            if norm(fl) > MAX_F {
                fl = scale(normalized(fl), MAX_F);
            }
            let fi = scale(add(fd, fl), 1.0 / 3.0);
            for &k in tri {
                let k = k as usize * 3;
                f[k] += fi[0];
                f[k + 1] += fi[1];
                f[k + 2] += fi[2];
            }
        }
    }

    /// Environment::Step: drag, then one projective-dynamics time step.
    pub fn step(&mut self) {
        if self.agar.is_some() {
            self.f_ext.iter_mut().for_each(|f| *f = 0.0);
        } else {
            self.drag();
        }
        let tan = if self.agar.is_some() { self.tangents(&self.x) } else { Vec::new() };
        let nv = self.nv;
        let inv_m = nv as f64 / MASS;
        // q = x + dt v + dt² M⁻¹ f
        let mut q = std::mem::take(&mut self.xk);
        for i in 0..nv * 3 {
            q[i] = self.x[i] + DT * self.v[i] + DT * DT * inv_m * self.f_ext[i];
        }
        let mut x = q.clone();
        let mut b = std::mem::take(&mut self.scratch);
        let mut it = 0;
        while it < MAX_ITER {
            // b = M/dt² q + Σ w Aᵀ p(x), in factor order
            for i in 0..nv {
                let r = self.perm[i] * 3;
                b[r] = self.m_dt2 * q[i * 3];
                b[r + 1] = self.m_dt2 * q[i * 3 + 1];
                b[r + 2] = self.m_dt2 * q[i * 3 + 2];
                if let Some(a) = self.agar {
                    let x0 = pt(&self.x, i as u32);
                    let t = tan[self.slice[i]];
                    let p = add(x0, scale(t, dot(t, sub(pt(&x, i as u32), x0))));
                    for c in 0..3 {
                        b[r + c] += a.w_t * x0[c] + (a.w_n - a.w_t) * p[c];
                    }
                }
            }
            for t in self.tets.iter_mut() {
                let xs = [pt(&x, t.idx[0]), pt(&x, t.idx[1]), pt(&x, t.idx[2]), pt(&x, t.idx[3])];
                let mut fm = [[0.0; 3]; 3];
                for k in 0..4 {
                    for r in 0..3 {
                        for c in 0..3 {
                            fm[r][c] += xs[k][r] * t.g[k][c];
                        }
                    }
                }
                let (u, s, v) = svd3_from(&fm, &mut t.guess);
                let mut rot = compose(&u, [1.0, 1.0, 1.0], &v);
                let mut pd = rot;
                if det(&fm) < 0.0 {
                    for r in 0..3 {
                        pd[r][2] = -rot[r][2];
                    }
                }
                // nearest F with det 1 (5 Newton steps on the singular values)
                let mut dd = [0.0; 3];
                for _ in 0..5 {
                    let cd = (s[0] + dd[0]) * (s[1] + dd[1]) * (s[2] + dd[2]) - 1.0;
                    let g = [(s[1] + dd[1]) * (s[2] + dd[2]), (s[0] + dd[0]) * (s[2] + dd[2]), (s[0] + dd[0]) * (s[1] + dd[1])];
                    let k = (dot(g, dd) - cd) / dot(g, g);
                    dd = scale(g, k);
                }
                let pv = compose(&u, add(s, dd), &v);
                for r in 0..3 {
                    for c in 0..3 {
                        rot[r][c] = t.w_dev * pd[r][c] + t.w_vol * pv[r][c];
                    }
                }
                for k in 0..4 {
                    let f = mv(&rot, t.g[k]);
                    let row = self.perm[t.idx[k] as usize] * 3;
                    b[row] += f[0];
                    b[row + 1] += f[1];
                    b[row + 2] += f[2];
                }
            }
            for fb in &self.fibers {
                let mut ff = [0.0; 3];
                for k in 0..4 {
                    ff = add(ff, scale(pt(&x, fb.idx[k]), fb.a[k]));
                }
                let p = scale(ff, (1.0 - self.act[fb.seg as usize]) * fb.w);
                for k in 0..4 {
                    let row = self.perm[fb.idx[k] as usize] * 3;
                    b[row] += fb.a[k] * p[0];
                    b[row + 1] += fb.a[k] * p[1];
                    b[row + 2] += fb.a[k] * p[2];
                }
            }
            self.chol.solve3(&mut b);
            let mut dx2 = 0.0;
            for i in 0..nv {
                let r = self.perm[i] * 3;
                for c in 0..3 {
                    let d = b[r + c] - x[i * 3 + c];
                    dx2 += d * d;
                }
            }
            it += 1;
            if dx2.sqrt() / (3 * nv) as f64 > EPS {
                for i in 0..nv {
                    let r = self.perm[i] * 3;
                    x[i * 3] = b[r];
                    x[i * 3 + 1] = b[r + 1];
                    x[i * 3 + 2] = b[r + 2];
                }
            } else {
                break;
            }
        }
        self.iterations = it;
        for i in 0..nv * 3 {
            self.v[i] = (x[i] - self.x[i]) / DT * DAMPING;
            self.x[i] = x[i];
        }
        self.xk = q;
        self.scratch = b;
    }
}
