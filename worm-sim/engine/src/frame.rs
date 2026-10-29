//! SoftCon's body frame (Worm::EvalBodyTransform + Environment::GetStates):
//! the rigid transform that best maps the rest mesh onto the current one, the
//! body-frame velocity averaged over the last 6 control steps, and the 17
//! sampled points' offsets from rest in that frame.
use crate::body::{det, svd3, Body, M3, V3};
use std::collections::VecDeque;

pub struct Frame {
    trans: VecDeque<V3>,
    pub rot: M3,
    pub t: V3,
}

impl Frame {
    pub fn new() -> Frame {
        Frame { trans: VecDeque::new(), rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], t: [0.0; 3] }
    }
    pub fn clear(&mut self) {
        self.trans.clear();
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

/// Kabsch fit of rest → current: (R, t)
pub fn fit(rest: &[f64], x: &[f64]) -> (M3, V3) {
    let n = rest.len() / 3;
    let mut mx = [0.0; 3];
    let mut my = [0.0; 3];
    for i in 0..n {
        for c in 0..3 {
            mx[c] += rest[i * 3 + c];
            my[c] += x[i * 3 + c];
        }
    }
    for c in 0..3 {
        mx[c] /= n as f64;
        my[c] /= n as f64;
    }
    let mut s = [[0.0; 3]; 3];
    for i in 0..n {
        for a in 0..3 {
            for b in 0..3 {
                s[a][b] += (rest[i * 3 + a] - mx[a]) * (x[i * 3 + b] - my[b]);
            }
        }
    }
    let (u, _sv, v) = svd3(&s);
    let flip = det(&[u[0], u[1], u[2]]) * det(&[v[0], v[1], v[2]]) < 0.0;
    let mut r = [[0.0; 3]; 3];
    for k in 0..3 {
        let sk = if flip && k == 2 { -1.0 } else { 1.0 };
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] += sk * v[k][i] * u[k][j];
            }
        }
    }
    let mut t = my;
    for i in 0..3 {
        for j in 0..3 {
            t[i] -= r[i][j] * mx[j];
        }
    }
    (r, t)
}

/// GetStates at the start of a control step (call once per step, 30 Hz):
/// [0,0,0, local average velocity (3), 17 × local offset (3), 17 × local velocity (3)]
pub fn body_frame(body: &Body, fr: &mut Frame) -> Vec<f64> {
    let (r, t) = fit(&body.rest, &body.x);
    fr.rot = r;
    fr.t = t;
    // local_body_rot = R · AngleAxis(π/2, y)
    let ay: M3 = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
    let mut lr = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                lr[i][j] += r[i][k] * ay[k][j];
            }
        }
    }
    let tr = |m: &M3, v: V3| -> V3 { [m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2], m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2], m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2]] };
    let mut avg = [0.0; 3];
    if fr.trans.len() > 1 {
        for i in 1..fr.trans.len() {
            for c in 0..3 {
                avg[c] += (fr.trans[i][c] - fr.trans[i - 1][c]) * 30.0;
            }
        }
        for c in 0..3 {
            avg[c] /= (fr.trans.len() - 1) as f64;
        }
    }
    let mut out = vec![0.0, 0.0, 0.0];
    out.extend_from_slice(&tr(&lr, avg));
    let ns = body.sampling.len();
    let mut vel = Vec::with_capacity(ns * 3);
    for &s in &body.sampling {
        let s = s as usize * 3;
        let p = [body.x[s] - t[0], body.x[s + 1] - t[1], body.x[s + 2] - t[2]];
        let lp = tr(&lr, p);
        let rp = [body.rest[s], body.rest[s + 1], body.rest[s + 2]];
        let rr = tr(&ay, rp);
        out.extend_from_slice(&[lp[0] - rr[0], lp[1] - rr[1], lp[2] - rr[2]]);
        vel.extend_from_slice(&tr(&lr, [body.v[s], body.v[s + 1], body.v[s + 2]]));
    }
    out.extend_from_slice(&vel);
    if fr.trans.len() > 6 {
        fr.trans.pop_front();
    }
    fr.trans.push_back(t);
    out
}
