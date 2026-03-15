use napi::bindgen_prelude::*;
use napi_derive::napi;
use realfft::RealFftPlanner;
use std::sync::{Arc, Mutex};

use crate::window::generate_window;

#[napi(object)]
#[derive(Clone)]
pub struct SpectrumResult {
    pub magnitudes_base64: String,
    pub magnitudes_length: u32,
    pub scale: f64,
    pub offset: f64,
    pub peak_frequency: f64,
    pub peak_magnitude: f64,
    pub average_magnitude: f64,
    pub dynamic_range: f64,
    pub frequency_resolution: f64,
    pub max_frequency: f64,
}

struct SpectrumAnalyzerInner {
    sample_rate: f64,
    fft_size: usize,
    window: Vec<f64>,
    target_sample_rate: Option<f64>,
}

impl SpectrumAnalyzerInner {
    fn analyze(&self, audio_data: &[f32]) -> napi::Result<SpectrumResult> {
        let fft_size = self.fft_size;
        let effective_sample_rate = self.target_sample_rate.unwrap_or(self.sample_rate);

        // Take up to fft_size samples
        let len = audio_data.len().min(fft_size);
        let mut input: Vec<f64> = Vec::with_capacity(fft_size);
        for i in 0..len {
            input.push(audio_data[i] as f64 * self.window[i.min(self.window.len() - 1)]);
        }
        // Zero-pad if needed
        input.resize(fft_size, 0.0);

        // Perform real FFT
        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut spectrum)
            .map_err(|e| napi::Error::from_reason(format!("FFT error: {}", e)))?;

        let num_bins = spectrum.len(); // fft_size/2 + 1
        let freq_resolution = effective_sample_rate / fft_size as f64;
        let max_frequency = effective_sample_rate / 2.0;

        // Compute magnitudes in dB
        let norm = 1.0 / fft_size as f64;
        let mut magnitudes_db: Vec<f64> = Vec::with_capacity(num_bins);
        let mut peak_mag = f64::NEG_INFINITY;
        let mut peak_idx = 0;
        let mut sum_mag = 0.0;
        let mut min_mag = f64::INFINITY;

        for (i, c) in spectrum.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt() * norm;
            let db = if mag > 1e-10 {
                20.0 * mag.log10()
            } else {
                -200.0
            };
            magnitudes_db.push(db);
            sum_mag += db;
            if db > peak_mag {
                peak_mag = db;
                peak_idx = i;
            }
            if db < min_mag {
                min_mag = db;
            }
        }

        let average_magnitude = sum_mag / num_bins as f64;
        let dynamic_range = peak_mag - min_mag;

        // Encode as Int16 base64
        let scale = 100.0; // dB * 100 → Int16 range
        let offset = 0.0;
        let mut int16_data: Vec<u8> = Vec::with_capacity(num_bins * 2);
        for &db in &magnitudes_db {
            let val = ((db - offset) * scale).clamp(-32768.0, 32767.0) as i16;
            int16_data.extend_from_slice(&val.to_le_bytes());
        }

        use base64::Engine;
        let magnitudes_base64 = base64::engine::general_purpose::STANDARD.encode(&int16_data);

        Ok(SpectrumResult {
            magnitudes_base64,
            magnitudes_length: num_bins as u32,
            scale,
            offset,
            peak_frequency: peak_idx as f64 * freq_resolution,
            peak_magnitude: peak_mag,
            average_magnitude,
            dynamic_range,
            frequency_resolution: freq_resolution,
            max_frequency,
        })
    }
}

#[napi]
pub struct SpectrumAnalyzer {
    inner: Arc<Mutex<SpectrumAnalyzerInner>>,
}

struct SpectrumAnalyzeTask {
    audio_data: Vec<f32>,
    inner: Arc<Mutex<SpectrumAnalyzerInner>>,
}

#[napi]
impl Task for SpectrumAnalyzeTask {
    type Output = SpectrumResult;
    type JsValue = SpectrumResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let inner = self.inner.lock().unwrap();
        inner.analyze(&self.audio_data)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
impl SpectrumAnalyzer {
    #[napi(constructor)]
    pub fn new(
        sample_rate: f64,
        fft_size: u32,
        window_function: Option<String>,
        target_sample_rate: Option<f64>,
    ) -> Result<Self> {
        let fft_size = fft_size as usize;
        let window_type = window_function.unwrap_or_else(|| "blackmanHarris".to_string());
        let window = generate_window(&window_type, fft_size)
            .map_err(|e| napi::Error::from(e))?;

        Ok(SpectrumAnalyzer {
            inner: Arc::new(Mutex::new(SpectrumAnalyzerInner {
                sample_rate,
                fft_size,
                window,
                target_sample_rate: target_sample_rate,
            })),
        })
    }

    #[napi]
    pub fn analyze(&self, audio_data: Float32Array) -> AsyncTask<SpectrumAnalyzeTask> {
        AsyncTask::new(SpectrumAnalyzeTask {
            audio_data: audio_data.to_vec(),
            inner: self.inner.clone(),
        })
    }
}
