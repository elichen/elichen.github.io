import test from "node:test";
import assert from "node:assert/strict";
import {
  prepareModalVoice,
  renderModalSamples,
  encodeMonoWav,
  ResonanceAudio,
} from "./audio.mjs";

function rms(samples, start, end) {
  let energy = 0;
  for (let index = start; index < end; index++) energy += samples[index] ** 2;
  return Math.sqrt(energy / (end - start));
}

test("a signed modal mixture renders finite, non-silent audio within its peak bound", () => {
  const options = {
    frequencies: [220, 317, 502, 910],
    amplitudes: [1, -0.8, 0.5, -0.2],
    gain: 0.9,
    decay: 0.6,
  };
  const samples = renderModalSamples(options, { sampleRate: 12000 });
  let peak = 0;
  for (const sample of samples) {
    assert.ok(Number.isFinite(sample));
    peak = Math.max(peak, Math.abs(sample));
  }
  assert.ok(peak > 0.05);
  assert.ok(peak <= 0.82 * options.gain + 1e-6);
  assert.equal(samples[0], 0);
  assert.equal(samples.at(-1), 0);
  assert.ok(rms(samples, 4800, 6000) < rms(samples, 120, 1320) * 0.03);
});

test("normalization preserves signed amplitude ratios, even for huge coefficients", () => {
  const voice = prepareModalVoice({
    frequencies: [200, 400, 600],
    amplitudes: [1e308, -5e307, 2.5e307],
  });
  assert.equal(voice.amplitudes[1] / voice.amplitudes[0], -0.5);
  assert.equal(voice.amplitudes[2] / voice.amplitudes[0], 0.25);
  assert.ok(voice.amplitudes.every(Number.isFinite));
  const opposite = renderModalSamples({
    frequencies: [440, 440],
    amplitudes: [1, -1],
    decay: 0.12,
  });
  assert.ok(opposite.every((value) => Math.abs(value) < 1e-7));
});

test("invalid and inaudible modes are suppressed; malformed input is safe silence", () => {
  const voice = prepareModalVoice(
    {
      frequencies: [-1, 0, NaN, Infinity, 4000, 4100, 3999, 440],
      amplitudes: [1, 1, 1, 1, 1, 1, -2, Infinity],
    },
    8000,
  );
  assert.deepEqual([...voice.frequencies], [3999]);
  assert.ok(voice.amplitudes[0] < 0);
  for (const options of [
    undefined,
    null,
    {},
    { frequencies: [NaN, -30] },
    { frequencies: [440], amplitudes: [0] },
    { frequencies: [440], gain: 0 },
  ]) {
    assert.ok(
      renderModalSamples(options, { sampleRate: 8000 }).every(
        (value) => value === 0,
      ),
    );
  }
});

test("frequency and brightness affect the actual signal and decay", () => {
  const options = { frequencies: [1000], amplitudes: [1], decay: 0.3, gain: 1 };
  const samples = renderModalSamples(options, { sampleRate: 16000 });
  let upwardCrossings = 0;
  for (let index = 161; index <= 1760; index++) {
    if (samples[index - 1] <= 0 && samples[index] > 0) upwardCrossings++;
  }
  assert.ok(Math.abs(upwardCrossings - 100) <= 1);
  const dark = prepareModalVoice({
    frequencies: [200, 1600],
    amplitudes: [1, 1],
    brightness: 0,
  });
  const bright = prepareModalVoice({
    frequencies: [200, 1600],
    amplitudes: [1, 1],
    brightness: 1,
  });
  assert.equal(dark.decays[0], bright.decays[0]);
  assert.ok(dark.decays[1] < bright.decays[1]);
  assert.deepEqual(dark.amplitudes, bright.amplitudes);
});

test("mode budget keeps strongest contributions and input ordering", () => {
  const frequencies = Array.from(
    { length: 40 },
    (_, index) => 100 + index * 10,
  );
  const amplitudes = frequencies.map((_, index) => index + 1);
  const voice = prepareModalVoice({ frequencies, amplitudes });
  assert.equal(voice.frequencies.length, 32);
  assert.equal(voice.frequencies[0], 180);
  assert.equal(voice.frequencies.at(-1), 490);
  assert.equal(amplitudes[0], 1);
});

test("WAV export has correct PCM metadata, length, and clipped finite encoding", async () => {
  const data = encodeMonoWav(new Float32Array([0, 1, -1, NaN, 2]), 16000);
  const view = new DataView(data);
  assert.equal(new TextDecoder().decode(new Uint8Array(data, 0, 4)), "RIFF");
  assert.equal(new TextDecoder().decode(new Uint8Array(data, 8, 4)), "WAVE");
  assert.equal(view.getUint32(24, true), 16000);
  assert.equal(view.getUint16(22, true), 1);
  assert.equal(view.getUint16(34, true), 16);
  assert.equal(view.getUint32(40, true), 10);
  assert.equal(view.getInt16(46, true), 32767);
  assert.equal(view.getInt16(48, true), -32768);
  assert.equal(view.getInt16(50, true), 0);
  const audio = new ResonanceAudio();
  assert.equal(audio.state, "locked");
  assert.equal(audio.strike({ frequencies: [440] }), null);
  audio.setVolume(0);
  const wav = audio.exportWav({ frequencies: [440], decay: 0.12 });
  assert.equal(wav.type, "audio/wav");
  assert.ok(
    new Uint8Array(await wav.arrayBuffer())
      .slice(44)
      .every((byte) => byte === 0),
  );
  const audibleExport = audio.exportWav(
    { frequencies: [440], decay: 0.12 },
    0.65,
  );
  assert.ok(
    new Uint8Array(await audibleExport.arrayBuffer())
      .slice(44)
      .some((byte) => byte !== 0),
  );
  assert.equal(
    audio.volume,
    0,
    "an explicit export level does not change muted live output",
  );
});
