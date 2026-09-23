// Every sound in the film is synthesized here with Web Audio. There are no audio files.

export class Sound {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.lastTink = 0;
  }

  get ready() {
    return !!this.ctx;
  }

  init() {
    if (this.ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    const c = (this.ctx = new AC());

    this.master = c.createGain();
    this.master.gain.value = this.muted ? 0 : 0.9;
    const comp = c.createDynamicsCompressor();
    comp.threshold.value = -16;
    comp.knee.value = 12;
    comp.ratio.value = 3.5;
    comp.attack.value = 0.01;
    comp.release.value = 0.3;
    this.master.connect(comp).connect(c.destination);

    this.verb = c.createConvolver();
    this.verb.buffer = this.impulse(3.6, 2.3);
    const wet = c.createGain();
    wet.gain.value = 0.55;
    this.verb.connect(wet).connect(this.master);

    const len = c.sampleRate * 2;
    this.noise = c.createBuffer(1, len, c.sampleRate);
    const d = this.noise.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;

    // The pad: four slowly gliding voices under a breathing low-pass filter.
    this.padFilter = c.createBiquadFilter();
    this.padFilter.type = 'lowpass';
    this.padFilter.frequency.value = 400;
    this.padFilter.Q.value = 0.6;
    this.padGain = c.createGain();
    this.padGain.gain.value = 0;
    this.padFilter.connect(this.padGain).connect(this.master);
    const send = c.createGain();
    send.gain.value = 0.8;
    this.padGain.connect(send).connect(this.verb);

    const lfo = c.createOscillator();
    lfo.frequency.value = 0.06;
    const lfoAmt = c.createGain();
    lfoAmt.gain.value = 160;
    lfo.connect(lfoAmt).connect(this.padFilter.frequency);
    lfo.start();

    this.voices = [];
    for (let i = 0; i < 4; i++) {
      const g = c.createGain();
      g.gain.value = 0.2;
      g.connect(this.padFilter);
      const trem = c.createOscillator();
      trem.frequency.value = 0.09 + i * 0.041;
      const tremAmt = c.createGain();
      tremAmt.gain.value = 0.07;
      trem.connect(tremAmt).connect(g.gain);
      trem.start();

      const o1 = c.createOscillator();
      o1.type = 'sine';
      const o2 = c.createOscillator();
      o2.type = 'triangle';
      o2.detune.value = 6 + i;
      const g2 = c.createGain();
      g2.gain.value = 0.45;
      o1.frequency.value = o2.frequency.value = 110;
      o1.connect(g);
      o2.connect(g2).connect(g);
      o1.start();
      o2.start();
      this.voices.push({ o1, o2 });
    }
  }

  impulse(seconds, decay) {
    const c = this.ctx;
    const len = Math.floor(c.sampleRate * seconds);
    const buf = c.createBuffer(2, len, c.sampleRate);
    for (let ch = 0; ch < 2; ch++) {
      const d = buf.getChannelData(ch);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, decay);
    }
    return buf;
  }

  now() {
    return this.ctx ? this.ctx.currentTime : 0;
  }

  resume() {
    if (this.ctx && this.ctx.state !== 'running') this.ctx.resume();
  }

  suspend() {
    if (this.ctx && this.ctx.state === 'running') this.ctx.suspend();
  }

  setMuted(m) {
    this.muted = m;
    if (this.ctx) this.master.gain.setTargetAtTime(m ? 0 : 0.9, this.ctx.currentTime, 0.08);
  }

  setChord(freqs, cutoff, level, glide = 1.5) {
    if (!this.ctx) return;
    const n = this.ctx.currentTime;
    this.voices.forEach((v, i) => {
      v.o1.frequency.setTargetAtTime(freqs[i], n, glide);
      v.o2.frequency.setTargetAtTime(freqs[i], n, glide);
    });
    this.padFilter.frequency.setTargetAtTime(cutoff, n, glide * 1.3);
    this.padGain.gain.setTargetAtTime(level, n, glide * 1.5);
  }

  // A gain node wired to the master bus, with an optional reverb send and pan.
  out(verb = 0.5, pan = 0) {
    const c = this.ctx;
    const g = c.createGain();
    let node = g;
    if (pan && c.createStereoPanner) {
      const p = c.createStereoPanner();
      p.pan.value = Math.max(-1, Math.min(1, pan));
      g.connect(p);
      node = p;
    }
    node.connect(this.master);
    if (verb > 0) {
      const s = c.createGain();
      s.gain.value = verb;
      node.connect(s).connect(this.verb);
    }
    return g;
  }

  pluck(freq, vol = 0.15, decay = 1.2, { verb = 0.6, pan = 0, delay = 0, type = 'sine' } = {}) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime + delay;
    const g = this.out(verb, pan);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t + decay);
    const o = c.createOscillator();
    o.type = type;
    o.frequency.value = freq;
    o.connect(g);
    const o2 = c.createOscillator();
    o2.frequency.value = freq * 2.01;
    const g2 = c.createGain();
    g2.gain.value = 0.16;
    o2.connect(g2).connect(g);
    o.start(t);
    o2.start(t);
    o.stop(t + decay + 0.05);
    o2.stop(t + decay + 0.05);
  }

  // Scattered, rate-limited glints for the thousands of fragments in the opening.
  tink(freq, pan) {
    if (!this.ctx) return;
    const n = this.ctx.currentTime;
    if (n - this.lastTink < 0.085 || Math.random() > 0.4) return;
    this.lastTink = n;
    this.pluck(freq, 0.022, 0.6, { verb: 0.9, pan });
  }

  bell(freq, vol = 0.1, decay = 3.5, { delay = 0, pan = 0 } = {}) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime + delay;
    const partials = [
      [1, 1, 1],
      [2.76, 0.38, 0.6],
      [5.4, 0.16, 0.35],
      [8.93, 0.06, 0.2],
    ];
    for (const [ratio, amp, dk] of partials) {
      const g = this.out(0.8, pan);
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vol * amp, t + 0.005);
      g.gain.exponentialRampToValueAtTime(0.0001, t + decay * dk);
      const o = c.createOscillator();
      o.frequency.value = freq * ratio;
      o.connect(g);
      o.start(t);
      o.stop(t + decay * dk + 0.05);
    }
  }

  click(vol = 0.3) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;
    const src = c.createBufferSource();
    src.buffer = this.noise;
    src.playbackRate.value = 0.8 + Math.random() * 0.4;
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.frequency.value = 2400 + Math.random() * 1600;
    bp.Q.value = 1.2;
    const g = this.out(0.15, (Math.random() - 0.5) * 0.3);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
    src.connect(bp).connect(g);
    src.start(t, Math.random() * 1.5, 0.08);

    const o = c.createOscillator();
    o.frequency.setValueAtTime(190, t);
    o.frequency.exponentialRampToValueAtTime(90, t + 0.03);
    const og = this.out(0);
    og.gain.setValueAtTime(vol * 0.5, t);
    og.gain.exponentialRampToValueAtTime(0.0001, t + 0.04);
    o.connect(og);
    o.start(t);
    o.stop(t + 0.05);
  }

  whoosh(dur, f0, f1, vol = 0.05) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;
    const src = c.createBufferSource();
    src.buffer = this.noise;
    src.loop = true;
    const bp = c.createBiquadFilter();
    bp.type = 'bandpass';
    bp.Q.value = 1.4;
    bp.frequency.setValueAtTime(f0, t);
    bp.frequency.exponentialRampToValueAtTime(f1, t + dur);
    const g = this.out(0.9);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + dur * 0.7);
    g.gain.linearRampToValueAtTime(0, t + dur);
    src.connect(bp).connect(g);
    src.start(t);
    src.stop(t + dur + 0.05);
  }

  // A soft, slightly sour "no" for a wrong guess.
  blip(freq = 110, vol = 0.14) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;
    const lp = c.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = 700;
    const g = this.out(0.3);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.45);
    lp.connect(g);
    for (const det of [-18, 18]) {
      const o = c.createOscillator();
      o.type = 'triangle';
      o.detune.value = det;
      o.frequency.setValueAtTime(freq, t);
      o.frequency.exponentialRampToValueAtTime(freq * 0.82, t + 0.35);
      o.connect(lp);
      o.start(t);
      o.stop(t + 0.5);
    }
  }

  boom(freq = 58, vol = 0.3, dur = 3) {
    if (!this.ctx) return;
    const c = this.ctx;
    const t = c.currentTime;
    const o = c.createOscillator();
    o.frequency.setValueAtTime(freq * 1.6, t);
    o.frequency.exponentialRampToValueAtTime(freq, t + 0.4);
    const g = this.out(0.5);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.03);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g);
    o.start(t);
    o.stop(t + dur + 0.05);
  }
}
