use audioadapter::Adapter;
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rubato::{
    Async, FixedAsync, PolynomialDegree, Resampler as RubatoResampler,
    SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::sync::{Arc, Mutex};

/// Resampler quality preset.
#[napi]
pub enum ResamplerQuality {
    /// Sinc, sinc_len=256, cubic interpolation, BlackmanHarris2
    Best = 0,
    /// Sinc, sinc_len=128, cubic, Blackman2
    High = 1,
    /// Sinc, sinc_len=64, linear, Hann
    Medium = 2,
    /// Polynomial cubic
    Low = 3,
    /// Polynomial linear
    Fastest = 4,
}

struct ResamplerInner {
    resampler: Box<dyn RubatoResampler<f32>>,
    remainder: Vec<f32>,
    channels: usize,
}

impl ResamplerInner {
    fn new(
        input_rate: f64,
        output_rate: f64,
        channels: usize,
        quality: ResamplerQuality,
    ) -> napi::Result<Self> {
        let ratio = output_rate / input_rate;
        let chunk_size = 1024;

        let resampler: Box<dyn RubatoResampler<f32>> = match quality {
            ResamplerQuality::Best => {
                let params = SincInterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    oversampling_factor: 256,
                    interpolation: SincInterpolationType::Cubic,
                    window: WindowFunction::BlackmanHarris2,
                };
                Box::new(
                    Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, channels, FixedAsync::Input)
                        .map_err(|e| napi::Error::from_reason(format!("Resampler error: {}", e)))?,
                )
            }
            ResamplerQuality::High => {
                let params = SincInterpolationParameters {
                    sinc_len: 128,
                    f_cutoff: 0.95,
                    oversampling_factor: 256,
                    interpolation: SincInterpolationType::Cubic,
                    window: WindowFunction::Blackman2,
                };
                Box::new(
                    Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, channels, FixedAsync::Input)
                        .map_err(|e| napi::Error::from_reason(format!("Resampler error: {}", e)))?,
                )
            }
            ResamplerQuality::Medium => {
                let params = SincInterpolationParameters {
                    sinc_len: 64,
                    f_cutoff: 0.95,
                    oversampling_factor: 128,
                    interpolation: SincInterpolationType::Linear,
                    window: WindowFunction::Hann,
                };
                Box::new(
                    Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, channels, FixedAsync::Input)
                        .map_err(|e| napi::Error::from_reason(format!("Resampler error: {}", e)))?,
                )
            }
            ResamplerQuality::Low => Box::new(
                Async::<f32>::new_poly(ratio, 1.0, PolynomialDegree::Cubic, chunk_size, channels, FixedAsync::Input)
                    .map_err(|e| napi::Error::from_reason(format!("Resampler error: {}", e)))?,
            ),
            ResamplerQuality::Fastest => Box::new(
                Async::<f32>::new_poly(ratio, 1.0, PolynomialDegree::Linear, chunk_size, channels, FixedAsync::Input)
                    .map_err(|e| napi::Error::from_reason(format!("Resampler error: {}", e)))?,
            ),
        };

        Ok(ResamplerInner {
            resampler,
            remainder: Vec::new(),
            channels,
        })
    }

    fn chunk_size(&self) -> usize {
        self.resampler.input_frames_next()
    }

    fn process(&mut self, input: &[f32]) -> napi::Result<Vec<f32>> {
        let channels = self.channels;
        let chunk_size = self.chunk_size();

        // Combine remainder with new input
        let mut data = Vec::with_capacity(self.remainder.len() + input.len());
        data.extend_from_slice(&self.remainder);
        data.extend_from_slice(input);
        self.remainder.clear();

        let total_frames = data.len() / channels;
        let mut output_all: Vec<f32> = Vec::new();
        let mut frame_offset = 0;

        while frame_offset + chunk_size <= total_frames {
            // Deinterleave one chunk into Vec<Vec<f32>>
            let mut chunk_channels: Vec<Vec<f32>> = (0..channels)
                .map(|_| vec![0.0f32; chunk_size])
                .collect();

            for f in 0..chunk_size {
                for ch in 0..channels {
                    chunk_channels[ch][f] = data[(frame_offset + f) * channels + ch];
                }
            }

            let input_adapter =
                SequentialSliceOfVecs::new(&chunk_channels, channels, chunk_size)
                    .map_err(|e| napi::Error::from_reason(format!("Adapter error: {}", e)))?;

            let out = self
                .resampler
                .process(&input_adapter, 0, None)
                .map_err(|e| napi::Error::from_reason(format!("Resample error: {}", e)))?;

            // Read output from InterleavedOwned using Adapter trait
            let out_frames = out.frames();
            let out_channels = out.channels();
            for f in 0..out_frames {
                for ch in 0..out_channels {
                    output_all.push(unsafe { out.read_sample_unchecked(ch, f) });
                }
            }

            frame_offset += chunk_size;
        }

        // Save remainder
        let remaining_samples = (total_frames - frame_offset) * channels;
        if remaining_samples > 0 {
            self.remainder.extend_from_slice(
                &data[frame_offset * channels..frame_offset * channels + remaining_samples],
            );
        }

        Ok(output_all)
    }

    fn flush(&mut self) -> napi::Result<Vec<f32>> {
        let channels = self.channels;
        let chunk_size = self.chunk_size();

        if self.remainder.is_empty() {
            return Ok(vec![]);
        }

        // Pad remainder to full chunk size
        let remainder_frames = self.remainder.len() / channels;

        let mut chunk_channels: Vec<Vec<f32>> = (0..channels)
            .map(|_| vec![0.0f32; chunk_size])
            .collect();

        for f in 0..remainder_frames {
            for ch in 0..channels {
                chunk_channels[ch][f] = self.remainder[f * channels + ch];
            }
        }

        self.remainder.clear();

        let input_adapter =
            SequentialSliceOfVecs::new(&chunk_channels, channels, chunk_size)
                .map_err(|e| napi::Error::from_reason(format!("Adapter error: {}", e)))?;

        let out = self
            .resampler
            .process(&input_adapter, 0, None)
            .map_err(|e| napi::Error::from_reason(format!("Resample error: {}", e)))?;

        let out_frames = out.frames();
        let out_channels = out.channels();
        let mut output = Vec::with_capacity(out_frames * out_channels);
        for f in 0..out_frames {
            for ch in 0..out_channels {
                output.push(unsafe { out.read_sample_unchecked(ch, f) });
            }
        }

        Ok(output)
    }

    fn output_delay(&self) -> usize {
        self.resampler.output_delay()
    }

    fn reset(&mut self) {
        self.resampler.reset();
        self.remainder.clear();
    }
}

#[napi]
pub struct Resampler {
    inner: Arc<Mutex<ResamplerInner>>,
    output_delay: u32,
}

// --- Process Task ---

struct ResamplerProcessTask {
    input: Vec<f32>,
    inner: Arc<Mutex<ResamplerInner>>,
}

#[napi]
impl Task for ResamplerProcessTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut inner = self.inner.lock().unwrap();
        inner.process(&self.input)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

// --- Flush Task ---

struct ResamplerFlushTask {
    inner: Arc<Mutex<ResamplerInner>>,
}

#[napi]
impl Task for ResamplerFlushTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut inner = self.inner.lock().unwrap();
        inner.flush()
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
impl Resampler {
    #[napi(constructor)]
    pub fn new(
        input_rate: f64,
        output_rate: f64,
        channels: Option<u32>,
        quality: Option<ResamplerQuality>,
    ) -> Result<Self> {
        let channels = channels.unwrap_or(1) as usize;
        let quality = quality.unwrap_or(ResamplerQuality::High);

        let inner = ResamplerInner::new(input_rate, output_rate, channels, quality)?;
        let output_delay = inner.output_delay() as u32;

        Ok(Resampler {
            inner: Arc::new(Mutex::new(inner)),
            output_delay,
        })
    }

    /// Stream processing (maintains filter state across calls).
    #[napi]
    pub fn process(&self, input: Float32Array) -> AsyncTask<ResamplerProcessTask> {
        AsyncTask::new(ResamplerProcessTask {
            input: input.to_vec(),
            inner: self.inner.clone(),
        })
    }

    /// Flush remaining samples from internal buffer.
    #[napi]
    pub fn flush(&self) -> AsyncTask<ResamplerFlushTask> {
        AsyncTask::new(ResamplerFlushTask {
            inner: self.inner.clone(),
        })
    }

    /// Reset filter state.
    #[napi]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.reset();
    }

    /// Output delay in frames.
    #[napi(getter)]
    pub fn output_delay(&self) -> u32 {
        self.output_delay
    }

    /// Release native resources.
    #[napi]
    pub fn dispose(&self) {
        // The Arc will be dropped when all references are gone.
    }
}

// --- One-shot resample function ---

struct ResampleOneShotTask {
    input: Vec<f32>,
    input_rate: f64,
    output_rate: f64,
    channels: usize,
    quality: ResamplerQuality,
}

#[napi]
impl Task for ResampleOneShotTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        if (self.input_rate - self.output_rate).abs() < 0.01 {
            return Ok(self.input.clone());
        }

        // Use the quality from the task (currently always High for one-shot)
        let q = match self.quality {
            ResamplerQuality::Best => ResamplerQuality::Best,
            ResamplerQuality::High => ResamplerQuality::High,
            ResamplerQuality::Medium => ResamplerQuality::Medium,
            ResamplerQuality::Low => ResamplerQuality::Low,
            ResamplerQuality::Fastest => ResamplerQuality::Fastest,
        };
        let mut inner = ResamplerInner::new(
            self.input_rate,
            self.output_rate,
            self.channels,
            q,
        )?;
        let mut result = inner.process(&self.input)?;
        let flush = inner.flush()?;
        result.extend_from_slice(&flush);
        Ok(result)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

/// One-shot resample (stateless, creates and destroys resampler internally).
#[napi]
pub fn resample(
    input: Float32Array,
    input_rate: f64,
    output_rate: f64,
    channels: Option<u32>,
    quality: Option<ResamplerQuality>,
) -> AsyncTask<ResampleOneShotTask> {
    AsyncTask::new(ResampleOneShotTask {
        input: input.to_vec(),
        input_rate,
        output_rate,
        channels: channels.unwrap_or(1) as usize,
        quality: quality.unwrap_or(ResamplerQuality::High),
    })
}
