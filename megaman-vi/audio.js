// Vim Blaster audio: a tiny WebAudio chiptune engine. No samples, no files --
// every sound is synthesized from oscillators and filtered noise. The context
// is created lazily on the first key so autoplay policies stay happy, and the
// music is a lookahead step sequencer that arpeggiates a per-stage chord loop.
const GameAudio = (() => {
    let ctx = null;
    let master = null;
    let musicGain = null;
    let sfxGain = null;
    let muted = false;
    let started = false;

    // Sequencer state.
    let timer = null;
    let stepTime = 0;
    let step = 0;
    let track = null;

    const A4 = 440;
    function midiToFreq(m) {
        return A4 * Math.pow(2, (m - 69) / 12);
    }

    // Chord helpers: a chord is a root MIDI note plus a quality (intervals).
    const MINOR = [0, 3, 7, 12];
    const MAJOR = [0, 4, 7, 12];
    function chord(root, quality) {
        return quality.map((i) => root + i);
    }

    // Per-stage music. Each track is a tempo, a chord progression (one chord
    // per bar of 16 sixteenth-notes), and waveform/voice character.
    const TRACKS = {
        menu: {
            bpm: 104, bass: 'triangle', lead: 'square', leadGain: 0.16, bassGain: 0.28, hat: 0.05,
            prog: [chord(45, MINOR), chord(41, MAJOR), chord(48, MAJOR), chord(43, MAJOR)]
        },
        foundry: {
            bpm: 142, bass: 'square', lead: 'square', leadGain: 0.15, bassGain: 0.3, hat: 0.07,
            prog: [chord(45, MINOR), chord(43, MAJOR), chord(41, MAJOR), chord(43, MAJOR)]
        },
        hydro: {
            bpm: 118, bass: 'triangle', lead: 'triangle', leadGain: 0.17, bassGain: 0.28, hat: 0.05,
            prog: [chord(50, MINOR), chord(46, MAJOR), chord(41, MAJOR), chord(48, MAJOR)]
        },
        sky: {
            bpm: 152, bass: 'square', lead: 'square', leadGain: 0.14, bassGain: 0.27, hat: 0.07,
            prog: [chord(48, MAJOR), chord(43, MAJOR), chord(45, MINOR), chord(41, MAJOR)]
        },
        final: {
            bpm: 136, bass: 'sawtooth', lead: 'square', leadGain: 0.15, bassGain: 0.3, hat: 0.06,
            prog: [chord(40, MINOR), chord(48, MAJOR), chord(45, MINOR), chord(47, MAJOR)]
        }
    };

    let noiseBuffer = null;
    function makeNoise() {
        const len = ctx.sampleRate * 0.5;
        noiseBuffer = ctx.createBuffer(1, len, ctx.sampleRate);
        const data = noiseBuffer.getChannelData(0);
        for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    }

    function init() {
        if (ctx) return;
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (!Ctor) return;
        ctx = new Ctor();
        master = ctx.createGain();
        master.gain.value = 0.85;
        master.connect(ctx.destination);
        musicGain = ctx.createGain();
        musicGain.gain.value = 0.55;
        musicGain.connect(master);
        sfxGain = ctx.createGain();
        sfxGain.gain.value = 0.8;
        sfxGain.connect(master);
        makeNoise();
    }

    // One enveloped oscillator note.
    function tone(time, freq, dur, type, gain, dest, glideTo, vibrato) {
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, time);
        if (glideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(1, glideTo), time + dur);
        if (vibrato) {
            const lfo = ctx.createOscillator();
            const lfoG = ctx.createGain();
            lfo.frequency.value = vibrato;
            lfoG.gain.value = freq * 0.03;
            lfo.connect(lfoG);
            lfoG.connect(osc.frequency);
            lfo.start(time);
            lfo.stop(time + dur);
        }
        g.gain.setValueAtTime(0.0001, time);
        g.gain.exponentialRampToValueAtTime(gain, time + 0.008);
        g.gain.exponentialRampToValueAtTime(0.0001, time + dur);
        osc.connect(g);
        g.connect(dest);
        osc.start(time);
        osc.stop(time + dur + 0.02);
    }

    // A filtered noise burst (hats, impacts, explosions).
    function noise(time, dur, gain, dest, filterType, filterFreq) {
        const src = ctx.createBufferSource();
        src.buffer = noiseBuffer;
        const g = ctx.createGain();
        const filter = ctx.createBiquadFilter();
        filter.type = filterType;
        filter.frequency.value = filterFreq;
        g.gain.setValueAtTime(gain, time);
        g.gain.exponentialRampToValueAtTime(0.0001, time + dur);
        src.connect(filter);
        filter.connect(g);
        g.connect(dest);
        src.start(time);
        src.stop(time + dur + 0.02);
    }

    function scheduleStep(s, time) {
        if (!track) return;
        const beat = s % 16;
        const bar = Math.floor(s / 16) % track.prog.length;
        const ch = track.prog[bar];
        const stepDur = 60 / track.bpm / 4;

        // Bass: root on the downbeats, fifth on the and-of-2.
        if (beat % 4 === 0) {
            tone(time, midiToFreq(ch[0] - 12), stepDur * 1.8, track.bass, track.bassGain, musicGain);
        } else if (beat === 10) {
            tone(time, midiToFreq(ch[2] - 12), stepDur * 1.4, track.bass, track.bassGain * 0.8, musicGain);
        }
        // Lead: a rolling arpeggio across the chord tones.
        const arp = [0, 1, 2, 3, 2, 1, 2, 3];
        if (beat % 2 === 0) {
            const note = ch[arp[(s / 1) % arp.length % arp.length] % ch.length];
            tone(time, midiToFreq(note + 12), stepDur * 1.1, track.lead, track.leadGain, musicGain);
        }
        // Hat: offbeat sixteenths.
        if (beat % 2 === 1) {
            noise(time, 0.03, track.hat, musicGain, 'highpass', 7000);
        }
        // Kick-ish thump on the beat.
        if (beat % 8 === 0) {
            tone(time, 90, 0.12, 'sine', 0.3, musicGain, 40);
        }
    }

    function scheduler() {
        if (!ctx || !track) return;
        while (stepTime < ctx.currentTime + 0.12) {
            scheduleStep(step, stepTime);
            stepTime += 60 / track.bpm / 4;
            step += 1;
        }
    }

    // --- Public API ---
    function resume() {
        init();
        if (ctx && ctx.state === 'suspended') ctx.resume();
        started = true;
    }

    function playMusic(id) {
        init();
        if (!ctx) return;
        const next = TRACKS[id] || TRACKS.menu;
        if (track === next && timer) return;
        track = next;
        step = 0;
        stepTime = ctx.currentTime + 0.05;
        if (!timer) timer = setInterval(scheduler, 25);
    }

    function stopMusic() {
        if (timer) {
            clearInterval(timer);
            timer = null;
        }
        track = null;
    }

    function sfx(name, variant) {
        if (!ctx || muted) return;
        const t = ctx.currentTime;
        const out = sfxGain;
        switch (name) {
            case 'shot': {
                if (variant === 'ember') tone(t, 520, 0.16, 'sawtooth', 0.3, out, 180, 18);
                else if (variant === 'tide') tone(t, 680, 0.14, 'triangle', 0.32, out, 320);
                else if (variant === 'storm') { tone(t, 300, 0.18, 'square', 0.28, out, 120); noise(t, 0.1, 0.12, out, 'bandpass', 1200); }
                else tone(t, 880, 0.09, 'square', 0.3, out, 360);
                break;
            }
            case 'jump':
                tone(t, 320, 0.14, 'square', 0.26, out, 720);
                break;
            case 'charge':
                tone(t, 200, 0.5, 'sawtooth', 0.16, out, 620);
                noise(t + 0.3, 0.18, 0.08, out, 'bandpass', 2200);
                break;
            case 'dash':
                noise(t, 0.16, 0.2, out, 'highpass', 1500);
                tone(t, 600, 0.14, 'sawtooth', 0.16, out, 200);
                break;
            case 'land':
                tone(t, 150, 0.1, 'sine', 0.3, out, 70);
                noise(t, 0.06, 0.12, out, 'lowpass', 800);
                break;
            case 'hit':
                tone(t, 200, 0.22, 'square', 0.32, out, 70);
                noise(t, 0.18, 0.22, out, 'bandpass', 600);
                break;
            case 'enemyHit':
                tone(t, 440, 0.05, 'square', 0.18, out, 300);
                break;
            case 'enemyDeath':
                noise(t, 0.2, 0.24, out, 'bandpass', 900);
                tone(t, 300, 0.18, 'square', 0.18, out, 120);
                break;
            case 'weak':
                tone(t, 1200, 0.2, 'square', 0.3, out, 380, 26);
                noise(t, 0.12, 0.16, out, 'highpass', 4000);
                break;
            case 'bossDeath':
                noise(t, 0.7, 0.34, out, 'lowpass', 1400);
                tone(t, 420, 0.7, 'sawtooth', 0.26, out, 60, 8);
                tone(t + 0.08, 300, 0.6, 'square', 0.2, out, 50);
                break;
            case 'clear': {
                const notes = [60, 64, 67, 72];
                notes.forEach((n, i) => tone(t + i * 0.11, midiToFreq(n), 0.18, 'square', 0.28, out));
                break;
            }
            case 'menuMove':
                tone(t, 660, 0.05, 'square', 0.2, out, 760);
                break;
            case 'menuSelect':
                tone(t, 520, 0.08, 'square', 0.26, out, 780);
                tone(t + 0.08, 780, 0.12, 'square', 0.26, out);
                break;
            default:
                break;
        }
    }

    function toggleMute() {
        muted = !muted;
        if (master) master.gain.value = muted ? 0 : 0.85;
        return muted;
    }

    function isMuted() {
        return muted;
    }

    // Pause audio when the tab is hidden; resume when it returns.
    document.addEventListener('visibilitychange', () => {
        if (!ctx) return;
        if (document.hidden) ctx.suspend();
        else if (started && !muted) ctx.resume();
    });

    return { resume, playMusic, stopMusic, sfx, toggleMute, isMuted };
})();
