use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::f64::consts::PI;

use crate::error::DspError;

/// Generate window function coefficients (synchronous, lightweight).
#[napi]
pub fn create_window(window_type: String, length: u32) -> Result<Float64Array> {
    let len = length as usize;
    let coeffs = generate_window(&window_type, len)
        .map_err(|e| napi::Error::from(e))?;
    Ok(Float64Array::new(coeffs))
}

struct ApplyWindowTask {
    signal: Vec<f32>,
    window_type: String,
}

#[napi]
impl Task for ApplyWindowTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let len = self.signal.len();
        let window = generate_window(&self.window_type, len)
            .map_err(|e| napi::Error::from(e))?;
        let mut output = self.signal.clone();
        for (i, s) in output.iter_mut().enumerate() {
            *s *= window[i] as f32;
        }
        Ok(output)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

/// Apply a window function to a signal.
#[napi]
pub fn apply_window(signal: Float32Array, window_type: String) -> AsyncTask<ApplyWindowTask> {
    AsyncTask::new(ApplyWindowTask {
        signal: signal.to_vec(),
        window_type,
    })
}

pub fn generate_window(window_type: &str, length: usize) -> std::result::Result<Vec<f64>, DspError> {
    if length == 0 {
        return Ok(vec![]);
    }
    if length == 1 {
        return Ok(vec![1.0]);
    }

    let nm1 = (length - 1) as f64;

    match window_type {
        "rectangular" => Ok(vec![1.0; length]),
        "hann" => Ok((0..length)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / nm1).cos()))
            .collect()),
        "hamming" => Ok((0..length)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / nm1).cos())
            .collect()),
        "blackman" => Ok((0..length)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / nm1;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect()),
        "blackmanHarris" => Ok((0..length)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / nm1;
                0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos()
                    - 0.01168 * (3.0 * x).cos()
            })
            .collect()),
        "kaiser" => {
            // Default beta = 5.0 (similar to Kaiser-Bessel derived)
            let beta = 5.0;
            let alpha = nm1 / 2.0;
            Ok((0..length)
                .map(|i| {
                    let r = (i as f64 - alpha) / alpha;
                    bessel_i0(beta * (1.0 - r * r).sqrt()) / bessel_i0(beta)
                })
                .collect())
        }
        "flatTop" => Ok((0..length)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / nm1;
                0.21557895 - 0.41663158 * x.cos() + 0.277263158 * (2.0 * x).cos()
                    - 0.083578947 * (3.0 * x).cos()
                    + 0.006947368 * (4.0 * x).cos()
            })
            .collect()),
        "bartlett" => Ok((0..length)
            .map(|i| 1.0 - ((i as f64 - nm1 / 2.0) / (nm1 / 2.0)).abs())
            .collect()),
        _ => Err(DspError(format!("Unknown window type: {}", window_type))),
    }
}

/// Modified Bessel function of the first kind, order 0 (I0)
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x_half = x / 2.0;
    for k in 1..50 {
        term *= (x_half / k as f64) * (x_half / k as f64);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}
