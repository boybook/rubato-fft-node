use napi::bindgen_prelude::*;
use napi_derive::napi;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

// --- Magnitude Spectrum ---

struct MagnitudeSpectrumTask {
    signal: Vec<f32>,
    fft_size: usize,
}

#[napi]
impl Task for MagnitudeSpectrumTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let fft_size = self.fft_size;
        let mut input: Vec<f64> = Vec::with_capacity(fft_size);
        for i in 0..fft_size {
            input.push(if i < self.signal.len() {
                self.signal[i] as f64
            } else {
                0.0
            });
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut spectrum)
            .map_err(|e| napi::Error::from_reason(format!("FFT error: {}", e)))?;

        let norm = 1.0 / fft_size as f64;
        Ok(spectrum
            .iter()
            .map(|c| ((c.re * c.re + c.im * c.im).sqrt() * norm) as f32)
            .collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
pub fn magnitude_spectrum(signal: Float32Array, fft_size: u32) -> AsyncTask<MagnitudeSpectrumTask> {
    AsyncTask::new(MagnitudeSpectrumTask {
        signal: signal.to_vec(),
        fft_size: fft_size as usize,
    })
}

// --- Power Spectrum dB ---

struct PowerSpectrumDbTask {
    signal: Vec<f32>,
    fft_size: usize,
}

#[napi]
impl Task for PowerSpectrumDbTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let fft_size = self.fft_size;
        let mut input: Vec<f64> = Vec::with_capacity(fft_size);
        for i in 0..fft_size {
            input.push(if i < self.signal.len() {
                self.signal[i] as f64
            } else {
                0.0
            });
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut spectrum)
            .map_err(|e| napi::Error::from_reason(format!("FFT error: {}", e)))?;

        let norm = 1.0 / (fft_size as f64 * fft_size as f64);
        Ok(spectrum
            .iter()
            .map(|c| {
                let power = (c.re * c.re + c.im * c.im) * norm;
                let db = if power > 1e-20 {
                    10.0 * power.log10()
                } else {
                    -200.0
                };
                db as f32
            })
            .collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
pub fn power_spectrum_db(signal: Float32Array, fft_size: u32) -> AsyncTask<PowerSpectrumDbTask> {
    AsyncTask::new(PowerSpectrumDbTask {
        signal: signal.to_vec(),
        fft_size: fft_size as usize,
    })
}

// --- Real FFT (complex output as interleaved [re, im, ...]) ---

struct RealFftTask {
    signal: Vec<f32>,
    fft_size: usize,
}

#[napi]
impl Task for RealFftTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let fft_size = self.fft_size;
        let mut input: Vec<f64> = Vec::with_capacity(fft_size);
        for i in 0..fft_size {
            input.push(if i < self.signal.len() {
                self.signal[i] as f64
            } else {
                0.0
            });
        }

        let mut planner = RealFftPlanner::<f64>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut input, &mut spectrum)
            .map_err(|e| napi::Error::from_reason(format!("FFT error: {}", e)))?;

        let mut output = Vec::with_capacity(spectrum.len() * 2);
        for c in &spectrum {
            output.push(c.re as f32);
            output.push(c.im as f32);
        }
        Ok(output)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi(js_name = "realFft")]
pub fn real_fft(signal: Float32Array, fft_size: u32) -> AsyncTask<RealFftTask> {
    AsyncTask::new(RealFftTask {
        signal: signal.to_vec(),
        fft_size: fft_size as usize,
    })
}

// --- Real IFFT ---

struct RealIfftTask {
    spectrum: Vec<f32>,
    fft_size: usize,
}

#[napi]
impl Task for RealIfftTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let fft_size = self.fft_size;
        let num_complex = fft_size / 2 + 1;

        if self.spectrum.len() < num_complex * 2 {
            return Err(napi::Error::from_reason(format!(
                "Spectrum length {} too short for fftSize {}, expected {} elements",
                self.spectrum.len(),
                fft_size,
                num_complex * 2
            )));
        }

        let mut spectrum: Vec<Complex<f64>> = (0..num_complex)
            .map(|i| Complex::new(self.spectrum[i * 2] as f64, self.spectrum[i * 2 + 1] as f64))
            .collect();

        let mut planner = RealFftPlanner::<f64>::new();
        let c2r = planner.plan_fft_inverse(fft_size);
        let mut output_f64 = c2r.make_output_vec();
        c2r.process(&mut spectrum, &mut output_f64)
            .map_err(|e| napi::Error::from_reason(format!("IFFT error: {}", e)))?;

        // Normalize (realfft doesn't normalize)
        let norm = 1.0 / fft_size as f64;
        Ok(output_f64.iter().map(|&v| (v * norm) as f32).collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi(js_name = "realIfft")]
pub fn real_ifft(spectrum: Float32Array, fft_size: u32) -> AsyncTask<RealIfftTask> {
    AsyncTask::new(RealIfftTask {
        spectrum: spectrum.to_vec(),
        fft_size: fft_size as usize,
    })
}
