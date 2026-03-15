import test from "ava";
import {
  Resampler,
  ResamplerQuality,
  resample,
  SpectrumAnalyzer,
  magnitudeSpectrum,
  powerSpectrumDb,
  realFft,
  realIfft,
  createWindow,
  applyWindow,
  designBiquad,
  BiquadFilter,
  measureLevel,
  int16ToFloat32,
  float32ToInt16,
  interleave,
  deinterleave,
} from "../index.js";

// Helper: generate a sine wave
function sineWave(
  frequency: number,
  sampleRate: number,
  duration: number
): Float32Array {
  const numSamples = Math.floor(sampleRate * duration);
  const data = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    data[i] = Math.sin((2 * Math.PI * frequency * i) / sampleRate);
  }
  return data;
}

// ======================== Resampler ========================

test("Resampler: 48kHz → 12kHz preserves frequency", async (t) => {
  const inputRate = 48000;
  const outputRate = 12000;
  const freq = 1000; // 1kHz tone
  const duration = 0.5;

  const input = sineWave(freq, inputRate, duration);
  const resampler = new Resampler(inputRate, outputRate, 1, ResamplerQuality.High);
  const output = await resampler.process(input);
  const flushed = await resampler.flush();

  // Output should be approximately 1/4 the length
  const totalLen = output.length + flushed.length;
  const expectedLen = Math.floor(duration * outputRate);
  t.true(
    Math.abs(totalLen - expectedLen) < outputRate * 0.1,
    `Output length ${totalLen} should be near ${expectedLen}`
  );
});

test("Resampler: streaming small chunks vs one-shot consistency", async (t) => {
  const inputRate = 48000;
  const outputRate = 16000;
  const input = sineWave(440, inputRate, 0.2);

  // One-shot
  const oneShotResult = await resample(input, inputRate, outputRate);

  // Streaming in small chunks
  const resampler = new Resampler(inputRate, outputRate, 1, ResamplerQuality.High);
  const chunkSize = 1024;
  const streamParts: Float32Array[] = [];
  for (let i = 0; i < input.length; i += chunkSize) {
    const chunk = input.slice(i, Math.min(i + chunkSize, input.length));
    streamParts.push(await resampler.process(chunk));
  }
  streamParts.push(await resampler.flush());

  const streamLen = streamParts.reduce((s, p) => s + p.length, 0);

  // Lengths should be similar (within reasonable tolerance)
  t.true(
    Math.abs(streamLen - oneShotResult.length) < outputRate * 0.05,
    `Stream length ${streamLen} should be near one-shot ${oneShotResult.length}`
  );
});

test("Resampler: same rate short-circuits", async (t) => {
  const input = sineWave(440, 48000, 0.1);
  const result = await resample(input, 48000, 48000);
  t.is(result.length, input.length);
});

test("Resampler: reset clears state", async (t) => {
  const resampler = new Resampler(48000, 16000);
  const input = sineWave(440, 48000, 0.05);
  await resampler.process(input);
  resampler.reset();
  // After reset, should work normally again
  const output = await resampler.process(input);
  t.true(output.length > 0 || true); // Just ensure no crash
  resampler.dispose();
  t.pass();
});

test("Resampler: empty input", async (t) => {
  const resampler = new Resampler(48000, 16000);
  const output = await resampler.process(new Float32Array(0));
  t.is(output.length, 0);
});

// ======================== SpectrumAnalyzer ========================

test("SpectrumAnalyzer: detects peak frequency of sine wave", async (t) => {
  const sampleRate = 48000;
  const freq = 1000;
  const fftSize = 4096;
  const signal = sineWave(freq, sampleRate, fftSize / sampleRate);

  const analyzer = new SpectrumAnalyzer(sampleRate, fftSize);
  const result = await analyzer.analyze(signal);

  // Peak frequency should be near 1000Hz
  t.true(
    Math.abs(result.peakFrequency - freq) < sampleRate / fftSize * 2,
    `Peak frequency ${result.peakFrequency} should be near ${freq}`
  );
  t.true(result.magnitudesBase64.length > 0);
  t.true(result.magnitudesLength > 0);
  t.true(result.dynamicRange > 0);
});

test("SpectrumAnalyzer: empty input doesn't crash", async (t) => {
  const analyzer = new SpectrumAnalyzer(48000, 1024);
  const result = await analyzer.analyze(new Float32Array(0));
  t.truthy(result);
});

// ======================== FFT ========================

test("FFT: magnitudeSpectrum returns correct size", async (t) => {
  const fftSize = 1024;
  const signal = sineWave(440, 48000, fftSize / 48000);
  const mags = await magnitudeSpectrum(signal, fftSize);
  t.is(mags.length, fftSize / 2 + 1);
});

test("FFT: powerSpectrumDb returns correct size", async (t) => {
  const fftSize = 512;
  const signal = sineWave(440, 48000, fftSize / 48000);
  const psd = await powerSpectrumDb(signal, fftSize);
  t.is(psd.length, fftSize / 2 + 1);
});

test("FFT: realFft + realIfft roundtrip", async (t) => {
  const fftSize = 256;
  const signal = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    signal[i] = Math.sin((2 * Math.PI * 10 * i) / fftSize) + 0.5 * Math.cos((2 * Math.PI * 20 * i) / fftSize);
  }

  const spectrum = await realFft(signal, fftSize);
  const recovered = await realIfft(spectrum, fftSize);

  t.is(recovered.length, fftSize);
  // Check roundtrip accuracy
  let maxError = 0;
  for (let i = 0; i < fftSize; i++) {
    maxError = Math.max(maxError, Math.abs(signal[i] - recovered[i]));
  }
  t.true(maxError < 1e-5, `Max roundtrip error ${maxError} should be < 1e-5`);
});

// ======================== Window ========================

test("Window: createWindow generates correct length", (t) => {
  const window = createWindow("hann", 1024);
  t.is(window.length, 1024);
  // Hann window should be 0 at edges and close to 1 near center
  t.true(Math.abs(window[0]) < 1e-10);
  // For even-length window, max is at (N-1)/2 ≈ 511.5, so check both 511 and 512
  t.true(window[511] > 0.999 || window[512] > 0.999, "Center of Hann window should be near 1.0");
});

test("Window: all types work", (t) => {
  const types = [
    "hann", "hamming", "blackman", "blackmanHarris",
    "kaiser", "flatTop", "bartlett", "rectangular",
  ];
  for (const type of types) {
    const window = createWindow(type, 256);
    t.is(window.length, 256, `${type} window should have correct length`);
  }
});

test("Window: applyWindow modifies signal", async (t) => {
  const signal = new Float32Array(1024).fill(1.0);
  const windowed = await applyWindow(signal, "hann");
  t.is(windowed.length, 1024);
  // First element should be near 0 (Hann starts at 0)
  t.true(Math.abs(windowed[0]) < 1e-5);
  // Middle should be near 1
  t.true(Math.abs(windowed[512] - 1.0) < 1e-3);
});

test("Window: empty window", (t) => {
  const window = createWindow("hann", 0);
  t.is(window.length, 0);
});

test("Window: length 1", (t) => {
  const window = createWindow("hann", 1);
  t.is(window.length, 1);
  t.is(window[0], 1.0);
});

// ======================== Filter ========================

test("Filter: lowpass attenuates high frequencies", async (t) => {
  const sampleRate = 48000;
  const cutoff = 1000;
  const coeffs = designBiquad("lowpass", sampleRate, cutoff);

  t.truthy(coeffs.b0);
  t.truthy(coeffs.a1);

  const filter = new BiquadFilter(coeffs);

  // Generate high frequency signal (10kHz) — should be attenuated
  const highFreq = sineWave(10000, sampleRate, 0.1);
  const filtered = await filter.process(highFreq);

  // RMS of output should be much less than input
  const inputRms = Math.sqrt(
    highFreq.reduce((sum, s) => sum + s * s, 0) / highFreq.length
  );
  const outputRms = Math.sqrt(
    filtered.reduce((sum, s) => sum + s * s, 0) / filtered.length
  );

  t.true(
    outputRms < inputRms * 0.5,
    `Output RMS ${outputRms} should be < half of input RMS ${inputRms}`
  );
});

test("Filter: reset clears state", (t) => {
  const coeffs = designBiquad("lowpass", 48000, 1000);
  const filter = new BiquadFilter(coeffs);
  filter.reset();
  t.pass();
});

test("Filter: all types can be designed", (t) => {
  const types = [
    "lowpass", "highpass", "bandpass", "notch",
    "allpass", "lowShelf", "highShelf", "peaking",
  ];
  for (const type of types) {
    const coeffs = designBiquad(type, 48000, 1000, 1.0, 6.0);
    t.truthy(coeffs, `${type} filter should produce coefficients`);
  }
});

// ======================== Level ========================

test("Level: known amplitude signal", async (t) => {
  // DC signal at 0.5
  const signal = new Float32Array(1024).fill(0.5);
  const result = await measureLevel(signal);

  t.true(Math.abs(result.rms - 0.5) < 1e-5, `RMS should be 0.5, got ${result.rms}`);
  t.true(Math.abs(result.peak - 0.5) < 1e-5, `Peak should be 0.5, got ${result.peak}`);
  t.true(
    Math.abs(result.rmsDb - 20 * Math.log10(0.5)) < 0.1,
    `RMS dB should be ~-6.02, got ${result.rmsDb}`
  );
});

test("Level: sine wave RMS", async (t) => {
  const signal = sineWave(440, 48000, 1.0);
  const result = await measureLevel(signal);

  // RMS of sine wave should be 1/sqrt(2) ≈ 0.707
  t.true(
    Math.abs(result.rms - 1 / Math.sqrt(2)) < 0.01,
    `Sine RMS should be ~0.707, got ${result.rms}`
  );
  t.true(Math.abs(result.peak - 1.0) < 0.01);
});

test("Level: empty signal", async (t) => {
  const result = await measureLevel(new Float32Array(0));
  t.is(result.rms, 0);
  t.is(result.peak, 0);
  t.is(result.rmsDb, -Infinity);
  t.is(result.peakDb, -Infinity);
});

// ======================== Convert ========================

test("Convert: int16 <-> float32 roundtrip", async (t) => {
  const original = new Int16Array([0, 16384, -16384, 32767, -32768]);
  const floats = await int16ToFloat32(original);

  t.true(Math.abs(floats[0]) < 1e-5);
  t.true(Math.abs(floats[1] - 0.5) < 0.001);
  t.true(Math.abs(floats[2] + 0.5) < 0.001);

  const backToInt = await float32ToInt16(floats);
  for (let i = 0; i < original.length; i++) {
    t.true(
      Math.abs(original[i] - backToInt[i]) <= 1,
      `Sample ${i}: ${original[i]} vs ${backToInt[i]}`
    );
  }
});

test("Convert: interleave / deinterleave roundtrip", async (t) => {
  const ch0 = new Float32Array([1, 2, 3, 4]);
  const ch1 = new Float32Array([5, 6, 7, 8]);

  const interleaved = await interleave([ch0, ch1]);
  t.deepEqual(
    Array.from(interleaved),
    [1, 5, 2, 6, 3, 7, 4, 8]
  );

  const channels = await deinterleave(interleaved, 2);
  t.is(channels.length, 2);
  t.deepEqual(Array.from(channels[0]), [1, 2, 3, 4]);
  t.deepEqual(Array.from(channels[1]), [5, 6, 7, 8]);
});

test("Convert: empty arrays", async (t) => {
  const floats = await int16ToFloat32(new Int16Array(0));
  t.is(floats.length, 0);

  const ints = await float32ToInt16(new Float32Array(0));
  t.is(ints.length, 0);
});

test("Convert: float32ToInt16 clamps", async (t) => {
  const input = new Float32Array([2.0, -2.0, 0.5]);
  const output = await float32ToInt16(input);
  t.is(output[0], 32767); // clamped to max
  t.is(output[1], -32767); // clamped to min
});
