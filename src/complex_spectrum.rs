use base64::Engine;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

use crate::window::generate_window;

#[napi(object)]
pub struct ComplexSpectrumAnalyzerOptions {
    pub sample_rate: f64,
    pub fft_size: u32,
    pub output_bins: u32,
    pub window_function: Option<String>,
    pub remove_dc: Option<bool>,
}

#[napi(object)]
#[derive(Clone)]
pub struct ComplexSpectrumResult {
    pub magnitudes_base64: String,
    pub magnitudes_length: u32,
    pub scale: f64,
    pub offset: f64,
    pub peak_offset_hz: f64,
    pub peak_magnitude: f64,
    pub average_magnitude: f64,
    pub dynamic_range: f64,
    pub frequency_resolution: f64,
    pub span_hz: f64,
}

struct ComplexSpectrumAnalyzerInner {
    sample_rate: f64,
    fft_size: usize,
    output_bins: usize,
    window: Vec<f64>,
    coherent_gain: f64,
    remove_dc: bool,
    fft: Arc<dyn Fft<f64>>,
}

impl ComplexSpectrumAnalyzerInner {
    fn analyze(&self, interleaved_iq: &[f32]) -> napi::Result<ComplexSpectrumResult> {
        if interleaved_iq.len() % 2 != 0 {
            return Err(napi::Error::from_reason(
                "Interleaved IQ input must contain an even number of scalar samples",
            ));
        }

        let available = interleaved_iq.len() / 2;
        let take = available.min(self.fft_size);
        let source_start = available.saturating_sub(take);
        let destination_start = self.fft_size - take;

        let (mean_i, mean_q) = if self.remove_dc && take > 0 {
            let mut sum_i = 0.0;
            let mut sum_q = 0.0;
            for index in source_start..available {
                sum_i += interleaved_iq[index * 2] as f64;
                sum_q += interleaved_iq[index * 2 + 1] as f64;
            }
            (sum_i / take as f64, sum_q / take as f64)
        } else {
            (0.0, 0.0)
        };

        let mut spectrum = vec![Complex::new(0.0, 0.0); self.fft_size];
        for offset in 0..take {
            let source = source_start + offset;
            let destination = destination_start + offset;
            let window = self.window[destination];
            spectrum[destination] = Complex::new(
                (interleaved_iq[source * 2] as f64 - mean_i) * window,
                (interleaved_iq[source * 2 + 1] as f64 - mean_q) * window,
            );
        }

        self.fft.process(&mut spectrum);

        let mut shifted_db = vec![0.0; self.fft_size];
        let mut full_peak = f64::NEG_INFINITY;
        let mut full_peak_index = 0usize;
        for (shifted_index, value) in shifted_db.iter_mut().enumerate() {
            let source_index = (shifted_index + self.fft_size / 2) % self.fft_size;
            let magnitude = spectrum[source_index].norm() / self.coherent_gain;
            let db = if magnitude > 1e-10 {
                20.0 * magnitude.log10()
            } else {
                -200.0
            };
            *value = db;
            if db > full_peak {
                full_peak = db;
                full_peak_index = shifted_index;
            }
        }

        let mut pooled = Vec::with_capacity(self.output_bins);
        for output_index in 0..self.output_bins {
            let start = output_index * self.fft_size / self.output_bins;
            let end = ((output_index + 1) * self.fft_size / self.output_bins).max(start + 1);
            let maximum = shifted_db[start..end]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            pooled.push(maximum);
        }

        let minimum = pooled.iter().copied().fold(f64::INFINITY, f64::min);
        let average = pooled.iter().sum::<f64>() / pooled.len() as f64;
        let scale = 0.01;
        let offset = 0.0;
        let mut encoded = Vec::with_capacity(pooled.len() * 2);
        for db in pooled {
            let quantized = ((db - offset) / scale).clamp(-32768.0, 32767.0) as i16;
            encoded.extend_from_slice(&quantized.to_le_bytes());
        }

        let resolution = self.sample_rate / self.fft_size as f64;
        let peak_offset_hz = -self.sample_rate / 2.0 + full_peak_index as f64 * resolution;
        Ok(ComplexSpectrumResult {
            magnitudes_base64: base64::engine::general_purpose::STANDARD.encode(encoded),
            magnitudes_length: self.output_bins as u32,
            scale,
            offset,
            peak_offset_hz,
            peak_magnitude: full_peak,
            average_magnitude: average,
            dynamic_range: full_peak - minimum,
            frequency_resolution: resolution,
            span_hz: self.sample_rate,
        })
    }
}

#[napi]
pub struct ComplexSpectrumAnalyzer {
    inner: Arc<ComplexSpectrumAnalyzerInner>,
}

struct ComplexSpectrumTask {
    interleaved_iq: Vec<f32>,
    inner: Arc<ComplexSpectrumAnalyzerInner>,
}

#[napi]
impl Task for ComplexSpectrumTask {
    type Output = ComplexSpectrumResult;
    type JsValue = ComplexSpectrumResult;

    fn compute(&mut self) -> Result<Self::Output> {
        self.inner.analyze(&self.interleaved_iq)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
impl ComplexSpectrumAnalyzer {
    #[napi(constructor)]
    pub fn new(options: ComplexSpectrumAnalyzerOptions) -> Result<Self> {
        let fft_size = options.fft_size as usize;
        let output_bins = options.output_bins as usize;
        if !options.sample_rate.is_finite() || options.sample_rate <= 0.0 {
            return Err(napi::Error::from_reason("sampleRate must be positive"));
        }
        if fft_size < 2 || !fft_size.is_power_of_two() {
            return Err(napi::Error::from_reason("fftSize must be a power of two"));
        }
        if output_bins == 0 || output_bins > fft_size {
            return Err(napi::Error::from_reason(
                "outputBins must be between 1 and fftSize",
            ));
        }

        let window_type = options
            .window_function
            .unwrap_or_else(|| "hann".to_string());
        let window = generate_window(&window_type, fft_size).map_err(napi::Error::from)?;
        let coherent_gain = window.iter().sum::<f64>().max(f64::EPSILON);
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);

        Ok(Self {
            inner: Arc::new(ComplexSpectrumAnalyzerInner {
                sample_rate: options.sample_rate,
                fft_size,
                output_bins,
                window,
                coherent_gain,
                remove_dc: options.remove_dc.unwrap_or(true),
                fft,
            }),
        })
    }

    #[napi]
    pub fn analyze(&self, interleaved_iq: Float32Array) -> AsyncTask<ComplexSpectrumTask> {
        AsyncTask::new(ComplexSpectrumTask {
            interleaved_iq: interleaved_iq.to_vec(),
            inner: self.inner.clone(),
        })
    }
}
