# rubato-fft-node

High-performance native audio DSP for Node.js, powered by Rust.

## Features

- **Resampling** — High-quality sample rate conversion via [Rubato](https://github.com/HEnquist/rubato), with configurable quality presets (sinc/polynomial)
- **FFT** — Real FFT, inverse FFT, magnitude/power spectrum via [RustFFT](https://github.com/ejmahler/RustFFT) + [realfft](https://github.com/HEnquist/realfft)
- **Window Functions** — Hann, Hamming, Blackman, Blackman-Harris, Kaiser, Flat Top, Bartlett, Rectangular
- **Biquad Filters** — Lowpass, highpass, bandpass, notch, allpass, shelving, peaking EQ
- **Level Metering** — RMS and peak measurement in linear and dBFS
- **Format Conversion** — Int16/Float32 conversion, interleave/deinterleave

All compute-intensive operations return **Promises** executed on the libuv thread pool via NAPI-RS AsyncTask, keeping the Node.js event loop free.

## Install

```bash
npm install rubato-fft-node
```

Prebuilt binaries are provided for:
- macOS ARM64 (Apple Silicon)
- macOS x64 (Intel)
- Linux x64 (glibc)
- Windows x64

No Rust toolchain required for installation.

## Quick Start

### Resampling

```js
import { Resampler, ResamplerQuality, resample } from 'rubato-fft-node';

// One-shot
const output = await resample(audioData, 48000, 16000);

// Streaming (maintains filter state across calls)
const resampler = new Resampler(48000, 16000, 1, ResamplerQuality.High);
const chunk1 = await resampler.process(inputChunk1);
const chunk2 = await resampler.process(inputChunk2);
const remaining = await resampler.flush();
resampler.dispose();
```

### FFT / Spectrum Analysis

```js
import { SpectrumAnalyzer, realFft, realIfft, magnitudeSpectrum } from 'rubato-fft-node';

// Spectrum analyzer with windowing
const analyzer = new SpectrumAnalyzer(48000, 4096, 'blackmanHarris');
const result = await analyzer.analyze(audioData);
console.log(`Peak: ${result.peakFrequency}Hz at ${result.peakMagnitude}dB`);

// Raw FFT
const spectrum = await realFft(signal, 1024);
const recovered = await realIfft(spectrum, 1024);

// Magnitude spectrum
const mags = await magnitudeSpectrum(signal, 1024);
```

### Window Functions

```js
import { createWindow, applyWindow } from 'rubato-fft-node';

const coefficients = createWindow('blackmanHarris', 1024); // sync
const windowed = await applyWindow(signal, 'hann');        // async
```

### Biquad Filters

```js
import { designBiquad, BiquadFilter } from 'rubato-fft-node';

const coeffs = designBiquad('lowpass', 48000, 1000, 0.707);
const filter = new BiquadFilter(coeffs);
const filtered = await filter.process(audioData);
filter.reset();
```

### Level Metering

```js
import { measureLevel } from 'rubato-fft-node';

const level = await measureLevel(audioData);
console.log(`RMS: ${level.rmsDb} dBFS, Peak: ${level.peakDb} dBFS`);
```

### Format Conversion

```js
import { int16ToFloat32, float32ToInt16, interleave, deinterleave } from 'rubato-fft-node';

const floats = await int16ToFloat32(int16Data);
const ints = await float32ToInt16(floatData);

const stereo = await interleave([leftChannel, rightChannel]);
const [left, right] = await deinterleave(stereoData, 2);
```

## API Reference

### Resampler

| API | Type | Description |
|-----|------|-------------|
| `new Resampler(inputRate, outputRate, channels?, quality?)` | Constructor | Create streaming resampler |
| `resampler.process(input)` | `Promise<Float32Array>` | Process audio chunk |
| `resampler.flush()` | `Promise<Float32Array>` | Flush remaining samples |
| `resampler.reset()` | `void` | Reset filter state |
| `resampler.outputDelay` | `number` | Output delay in frames |
| `resampler.dispose()` | `void` | Release native resources |
| `resample(input, inRate, outRate, ch?, quality?)` | `Promise<Float32Array>` | One-shot resample |

### Quality Presets

| Preset | Method | Description |
|--------|--------|-------------|
| `Best` | Sinc (len=256, cubic) | Highest quality, highest CPU |
| `High` | Sinc (len=128, cubic) | Good balance (default) |
| `Medium` | Sinc (len=64, linear) | Moderate quality |
| `Low` | Polynomial cubic | Fast |
| `Fastest` | Polynomial linear | Fastest, lowest quality |

### FFT / Spectrum

| API | Type | Description |
|-----|------|-------------|
| `new SpectrumAnalyzer(sampleRate, fftSize, window?, targetRate?)` | Constructor | Create analyzer |
| `analyzer.analyze(audioData)` | `Promise<SpectrumResult>` | Analyze audio |
| `magnitudeSpectrum(signal, fftSize)` | `Promise<Float32Array>` | Magnitude spectrum |
| `powerSpectrumDb(signal, fftSize)` | `Promise<Float32Array>` | Power spectrum (dB) |
| `realFft(signal, fftSize)` | `Promise<Float32Array>` | Forward real FFT |
| `realIfft(spectrum, fftSize)` | `Promise<Float32Array>` | Inverse real FFT |

### Window / Filter / Level / Convert

| API | Type | Description |
|-----|------|-------------|
| `createWindow(type, length)` | `Float64Array` | Generate window coefficients (sync) |
| `applyWindow(signal, type)` | `Promise<Float32Array>` | Apply window function |
| `designBiquad(type, rate, freq, q?, gain?)` | `BiquadCoefficients` | Design filter (sync) |
| `new BiquadFilter(coefficients)` | Constructor | Create stateful filter |
| `filter.process(input)` | `Promise<Float32Array>` | Apply filter |
| `measureLevel(samples)` | `Promise<LevelResult>` | Measure RMS/peak level |
| `int16ToFloat32(input)` | `Promise<Float32Array>` | Convert Int16 → Float32 |
| `float32ToInt16(input)` | `Promise<Int16Array>` | Convert Float32 → Int16 |
| `interleave(channels)` | `Promise<Float32Array>` | Interleave channels |
| `deinterleave(data, numCh)` | `Promise<Float32Array[]>` | Deinterleave channels |

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| macOS | ARM64 (Apple Silicon) | ✅ |
| macOS | x64 (Intel) | ✅ |
| Linux | x64 (glibc) | ✅ |
| Windows | x64 | ✅ |

## License

MIT
