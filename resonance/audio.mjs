/**
 * Browser modal percussion, without dependencies.
 *
 * Call await audio.unlock() from a user gesture, then audio.strike(options).
 * Options: { frequencies, amplitudes, decay = 2.5, brightness = 0.6, gain = 0.65 }.
 * Frequencies are Hz; signed amplitudes preserve the solver's modal phase.
 * decay is the lowest mode's approximate time to -60 dB, in seconds. Brightness
 * controls how much faster upper modes decay, without changing their onset mix.
 *
 * prepareModalVoice() exposes the exact filtered, normalized playback data.
 * renderModalSamples() and encodeMonoWav() are pure, usable without Web Audio.
 */

const MAX_MODES = 32;
const MAX_VOICES = 8;
const HEADROOM = 0.82;
const ATTACK = 0.003;
const TAIL = 0.025;
const LOG_1000 = Math.log(1000);
const DEFAULT_SAMPLE_RATE = 44100;

function bounded(value, fallback, minimum, maximum) {
  return Number.isFinite(value)
    ? Math.max(minimum, Math.min(maximum, value))
    : fallback;
}

function validSampleRate(sampleRate) {
  return Math.round(bounded(sampleRate, DEFAULT_SAMPLE_RATE, 8000, 192000));
}

/**
 * Keep up to 32 strongest finite modes below Nyquist. amplitudes has L1 norm
 * HEADROOM * gain, so even perfectly aligned modes cannot exceed that peak.
 * Input arrays are never mutated. Empty/invalid mode sets are silent.
 */
export function prepareModalVoice(
  options = {},
  sampleRate = DEFAULT_SAMPLE_RATE,
) {
  options = options && typeof options === "object" ? options : {};
  sampleRate = validSampleRate(sampleRate);
  const decay = bounded(options.decay, 2.5, 0.12, 12);
  const brightness = bounded(options.brightness, 0.6, 0, 1);
  const gain = bounded(options.gain, 0.65, 0, 1);
  const sourceFrequencies = options.frequencies;
  const sourceAmplitudes = options.amplitudes;
  const candidates = [];
  const count = Math.min(sourceFrequencies?.length || 0, 4096);
  for (let index = 0; index < count; index++) {
    const frequency = sourceFrequencies[index];
    const amplitude = sourceAmplitudes == null ? 1 : sourceAmplitudes[index];
    if (
      !Number.isFinite(frequency) ||
      frequency <= 0 ||
      frequency >= sampleRate / 2
    )
      continue;
    if (!Number.isFinite(amplitude) || amplitude === 0) continue;
    candidates.push({ frequency, amplitude, index });
  }
  candidates.sort((a, b) => Math.abs(b.amplitude) - Math.abs(a.amplitude));
  const selected = candidates
    .slice(0, MAX_MODES)
    .sort((a, b) => a.index - b.index);
  // Scaling before summation also handles very large finite input coefficients.
  let largest = 0;
  let baseFrequency = Infinity;
  for (const mode of selected) {
    largest = Math.max(largest, Math.abs(mode.amplitude));
    baseFrequency = Math.min(baseFrequency, mode.frequency);
  }
  let sum = 0;
  for (const mode of selected) sum += Math.abs(mode.amplitude) / largest;
  const frequencies = new Float64Array(selected.length);
  const amplitudes = new Float64Array(selected.length);
  const decays = new Float64Array(selected.length);
  for (let index = 0; index < selected.length; index++) {
    const mode = selected[index];
    frequencies[index] = mode.frequency;
    amplitudes[index] = (mode.amplitude / largest / sum) * HEADROOM * gain;
    const ratio = Math.min(64, mode.frequency / baseFrequency);
    decays[index] = decay / (1 + (1 - brightness) * 1.8 * Math.sqrt(ratio - 1));
  }
  return {
    frequencies,
    amplitudes,
    decays,
    sampleRate,
    decay,
    brightness,
    gain,
    attack: ATTACK,
    tail: TAIL,
    duration: ATTACK + decay + TAIL,
  };
}

function envelopeAt(time, voice, modeIndex) {
  if (time <= 0 || time >= voice.duration) return 0;
  if (time < voice.attack) return time / voice.attack;
  const releaseTime = voice.attack + voice.decay;
  if (time <= releaseTime) {
    return Math.exp(
      (-LOG_1000 * (time - voice.attack)) / voice.decays[modeIndex],
    );
  }
  const releaseLevel = Math.exp(
    (-LOG_1000 * voice.decay) / voice.decays[modeIndex],
  );
  return (releaseLevel * (voice.duration - time)) / voice.tail;
}

/** Render exactly one strike. Duration includes the final fade to digital zero. */
export function renderModalSamples(options = {}, settings = {}) {
  const voice = prepareModalVoice(options, settings.sampleRate);
  const samples = new Float32Array(
    Math.ceil(voice.duration * voice.sampleRate) + 1,
  );
  for (let mode = 0; mode < voice.frequencies.length; mode++) {
    const amplitude = voice.amplitudes[mode];
    if (amplitude === 0) continue;
    const angularFrequency = 2 * Math.PI * voice.frequencies[mode];
    for (let frame = 1; frame < samples.length - 1; frame++) {
      const time = frame / voice.sampleRate;
      samples[frame] +=
        amplitude *
        Math.sin(angularFrequency * time) *
        envelopeAt(time, voice, mode);
    }
  }
  // Roundoff safety only; L1 normalization already bounds the mathematical sum.
  for (let frame = 0; frame < samples.length; frame++) {
    samples[frame] = Math.max(-1, Math.min(1, samples[frame]));
  }
  return samples;
}

/** Encode 16-bit PCM mono, including a standard RIFF/WAVE header. */
export function encodeMonoWav(samples, sampleRate = DEFAULT_SAMPLE_RATE) {
  sampleRate = validSampleRate(sampleRate);
  const length = samples?.length || 0;
  const buffer = new ArrayBuffer(44 + length * 2);
  const view = new DataView(buffer);
  function text(offset, value) {
    for (let index = 0; index < value.length; index++)
      view.setUint8(offset + index, value.charCodeAt(index));
  }
  text(0, "RIFF");
  view.setUint32(4, buffer.byteLength - 8, true);
  text(8, "WAVE");
  text(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, length * 2, true);
  for (let index = 0; index < length; index++) {
    const sample = Number.isFinite(samples[index])
      ? Math.max(-1, Math.min(1, samples[index]))
      : 0;
    view.setInt16(
      44 + index * 2,
      Math.round(sample * (sample < 0 ? 32768 : 32767)),
      true,
    );
  }
  return buffer;
}

export class ResonanceAudio {
  constructor() {
    this.context = null;
    this.master = null;
    this.limiter = null;
    this.analyser = null;
    this.analysisSamples = null;
    this.volume = 0.8;
    this.voices = new Set();
  }

  get state() {
    return this.context?.state || "locked";
  }

  async unlock() {
    if (!this.context || this.context.state === "closed") {
      const Context = globalThis.AudioContext || globalThis.webkitAudioContext;
      if (!Context)
        throw new Error("Web Audio is unavailable in this browser.");
      this.context = new Context();
      this.master = this.context.createGain();
      this.master.gain.value = this.volume;
      this.limiter = this.context.createDynamicsCompressor();
      this.limiter.threshold.value = -4;
      this.limiter.knee.value = 4;
      this.limiter.ratio.value = 20;
      this.limiter.attack.value = 0.002;
      this.limiter.release.value = 0.12;
      this.analyser = this.context.createAnalyser();
      this.analyser.fftSize = 1024;
      this.analysisSamples = new Float32Array(this.analyser.fftSize);
      this.master.connect(this.limiter);
      this.limiter.connect(this.analyser);
      this.analyser.connect(this.context.destination);
    }
    if (this.context.state !== "running") await this.context.resume();
    return this.context.state;
  }

  /** Peak of the actual output waveform; reuses its sample buffer between reads. */
  readPeak() {
    if (!this.analyser || this.context?.state !== "running") return 0;
    if (this.analysisSamples.length !== this.analyser.fftSize) {
      this.analysisSamples = new Float32Array(this.analyser.fftSize);
    }
    this.analyser.getFloatTimeDomainData(this.analysisSamples);
    let peak = 0;
    for (const sample of this.analysisSamples) {
      if (Number.isFinite(sample)) peak = Math.max(peak, Math.abs(sample));
    }
    return peak;
  }

  setVolume(volume) {
    this.volume = bounded(volume, this.volume, 0, 1);
    if (this.master) {
      const now = this.context.currentTime;
      this.master.gain.cancelScheduledValues(now);
      this.master.gain.setTargetAtTime(this.volume, now, 0.012);
    }
  }

  /** Returns prepared playback data, or null if locked or the strike is silent. */
  strike(options = {}) {
    const context = this.context;
    if (!context || context.state !== "running") return null;
    const prepared = prepareModalVoice(options, context.sampleRate);
    if (!prepared.frequencies.length || prepared.gain === 0) return null;
    while (this.voices.size >= MAX_VOICES)
      this.releaseVoice(this.voices.values().next().value);
    const now = context.currentTime + 0.002;
    const voice = {
      bus: context.createGain(),
      nodes: [],
      stopAt: now + prepared.duration,
      stopped: false,
    };
    voice.bus.connect(this.master);
    this.voices.add(voice);
    for (let index = 0; index < prepared.frequencies.length; index++) {
      const oscillator = context.createOscillator();
      const envelope = context.createGain();
      const amplitude = prepared.amplitudes[index];
      const tau = prepared.decays[index] / LOG_1000;
      oscillator.type = "sine";
      oscillator.frequency.value = prepared.frequencies[index];
      envelope.gain.setValueAtTime(0, now);
      envelope.gain.linearRampToValueAtTime(amplitude, now + prepared.attack);
      envelope.gain.setTargetAtTime(0, now + prepared.attack, tau);
      const releaseAt = now + prepared.attack + prepared.decay;
      const releaseLevel =
        amplitude *
        Math.exp((-LOG_1000 * prepared.decay) / prepared.decays[index]);
      envelope.gain.setValueAtTime(releaseLevel, releaseAt);
      envelope.gain.linearRampToValueAtTime(0, voice.stopAt);
      oscillator.connect(envelope);
      envelope.connect(voice.bus);
      voice.nodes.push({ oscillator, envelope });
      oscillator.start(now);
      oscillator.stop(voice.stopAt + 0.005);
      oscillator.onended = () => {
        oscillator.disconnect();
        envelope.disconnect();
        voice.ended = (voice.ended || 0) + 1;
        if (voice.ended === voice.nodes.length) {
          voice.bus.disconnect();
          this.voices.delete(voice);
        }
      };
    }
    return prepared;
  }

  releaseVoice(voice) {
    if (!voice || voice.stopped) return;
    voice.stopped = true;
    const now = this.context.currentTime;
    voice.bus.gain.cancelScheduledValues(now);
    voice.bus.gain.setValueAtTime(1, now);
    voice.bus.gain.linearRampToValueAtTime(0, now + 0.018);
    for (const { oscillator } of voice.nodes) {
      try {
        oscillator.stop(now + 0.022);
      } catch {
        /* The voice may have just ended. */
      }
    }
    this.voices.delete(voice);
  }

  stop() {
    for (const voice of Array.from(this.voices)) this.releaseVoice(voice);
  }

  /** A dry strike at current or explicit volume, independent of live playback. */
  exportWav(options = {}, volume = this.volume) {
    const gain =
      bounded(options?.gain, 0.65, 0, 1) * bounded(volume, this.volume, 0, 1);
    const samples = renderModalSamples(
      { ...options, gain },
      { sampleRate: DEFAULT_SAMPLE_RATE },
    );
    return new Blob([encodeMonoWav(samples, DEFAULT_SAMPLE_RATE)], {
      type: "audio/wav",
    });
  }
}
