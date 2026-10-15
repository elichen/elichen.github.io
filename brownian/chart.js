// Log-log chart of the beads' mean squared displacement against time lag, with
// the two limits: ballistic (2 kT/M · t², no fitting) and diffusive (4 D t).

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = e => (e === 0 ? '1' : e === 1 ? '10' : '10' + String(e).split('').map(c => SUP[c]).join(''));

export class MSDChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.colors = null;
  }

  draw({ points, T, M, D, tMin, tMax }) {
    const cv = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const c = this.colors;
    const pad = { l: 46, r: 12, t: 10, b: 30 };
    const x0 = Math.floor(Math.log10(tMin)), x1 = Math.ceil(Math.log10(tMax));
    const ys = points.map(p => Math.log10(p.msd));
    let y0 = Math.min(-3, ...ys.map(Math.floor)), y1 = Math.max(2, ...ys.map(Math.ceil));
    const X = lt => pad.l + (lt - x0) / (x1 - x0) * (w - pad.l - pad.r);
    const Y = ly => h - pad.b - (ly - y0) / (y1 - y0) * (h - pad.t - pad.b);

    g.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    g.lineWidth = 1;
    g.strokeStyle = c.grid;
    g.fillStyle = c.text3;
    g.textAlign = 'center';
    g.textBaseline = 'top';
    for (let e = x0; e <= x1; e++) {
      g.beginPath(); g.moveTo(X(e), pad.t); g.lineTo(X(e), h - pad.b); g.stroke();
      g.fillText(pow10(e), X(e), h - pad.b + 5);
    }
    g.textAlign = 'right';
    g.textBaseline = 'middle';
    const ystep = y1 - y0 > 6 ? 2 : 1;
    for (let e = y0; e <= y1; e += ystep) {
      g.beginPath(); g.moveTo(pad.l, Y(e)); g.lineTo(w - pad.r, Y(e)); g.stroke();
      g.fillText(pow10(e), pad.l - 6, Y(e));
    }
    g.textAlign = 'left';
    g.textBaseline = 'bottom';
    g.fillText('lag (τ)', w - pad.r - 44, h - pad.b - 3);

    g.save();
    g.beginPath();
    g.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
    g.clip();
    const line = (f, color, dash) => {
      g.strokeStyle = color;
      g.setLineDash(dash);
      g.lineWidth = 1.5;
      g.beginPath();
      for (let i = 0; i <= 60; i++) {
        const lt = x0 + (x1 - x0) * i / 60;
        const v = f(10 ** lt);
        if (i) g.lineTo(X(lt), Y(Math.log10(v))); else g.moveTo(X(lt), Y(Math.log10(v)));
      }
      g.stroke();
      g.setLineDash([]);
    };
    line(t => 2 * T / M * t * t, c.ballistic, [5, 4]);
    if (D > 0) line(t => 4 * D * t, c.diffusive, [2, 3]);
    g.fillStyle = c.series;
    for (const p of points) {
      g.beginPath();
      g.arc(X(Math.log10(p.t)), Y(Math.log10(p.msd)), 2.4, 0, 2 * Math.PI);
      g.fill();
    }
    g.restore();
  }
}
