use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

#[napi(object)]
#[derive(Clone)]
pub struct BiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

/// Design biquad filter coefficients (synchronous).
#[napi]
pub fn design_biquad(
    filter_type: String,
    sample_rate: f64,
    frequency: f64,
    q: Option<f64>,
    gain: Option<f64>,
) -> Result<BiquadCoefficients> {
    let q = q.unwrap_or(0.7071); // 1/sqrt(2)
    let gain = gain.unwrap_or(0.0);

    let w0 = 2.0 * PI * frequency / sample_rate;
    let cos_w0 = w0.cos();
    let sin_w0 = w0.sin();
    let alpha = sin_w0 / (2.0 * q);

    let (b0, b1, b2, a0, a1, a2) = match filter_type.as_str() {
        "lowpass" => (
            (1.0 - cos_w0) / 2.0,
            1.0 - cos_w0,
            (1.0 - cos_w0) / 2.0,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        ),
        "highpass" => (
            (1.0 + cos_w0) / 2.0,
            -(1.0 + cos_w0),
            (1.0 + cos_w0) / 2.0,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        ),
        "bandpass" => (
            alpha,
            0.0,
            -alpha,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        ),
        "notch" => (
            1.0,
            -2.0 * cos_w0,
            1.0,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        ),
        "allpass" => (
            1.0 - alpha,
            -2.0 * cos_w0,
            1.0 + alpha,
            1.0 + alpha,
            -2.0 * cos_w0,
            1.0 - alpha,
        ),
        "lowShelf" => {
            let a = 10.0_f64.powf(gain / 40.0);
            let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
            (
                a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha),
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0),
                a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0),
                (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha,
            )
        }
        "highShelf" => {
            let a = 10.0_f64.powf(gain / 40.0);
            let two_sqrt_a_alpha = 2.0 * a.sqrt() * alpha;
            (
                a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0),
                a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha),
                (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha,
                2.0 * ((a - 1.0) - (a + 1.0) * cos_w0),
                (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha,
            )
        }
        "peaking" => {
            let a = 10.0_f64.powf(gain / 40.0);
            (
                1.0 + alpha * a,
                -2.0 * cos_w0,
                1.0 - alpha * a,
                1.0 + alpha / a,
                -2.0 * cos_w0,
                1.0 - alpha / a,
            )
        }
        _ => return Err(napi::Error::from_reason(format!("Unknown filter type: {}", filter_type))),
    };

    Ok(BiquadCoefficients {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    })
}

struct BiquadFilterInner {
    coeffs: BiquadCoefficients,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl BiquadFilterInner {
    fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(input.len());
        for &sample in input {
            let x = sample as f64;
            let y = self.coeffs.b0 * x + self.coeffs.b1 * self.x1 + self.coeffs.b2 * self.x2
                - self.coeffs.a1 * self.y1
                - self.coeffs.a2 * self.y2;
            self.x2 = self.x1;
            self.x1 = x;
            self.y2 = self.y1;
            self.y1 = y;
            output.push(y as f32);
        }
        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[napi]
pub struct BiquadFilter {
    inner: Arc<Mutex<BiquadFilterInner>>,
}

struct BiquadProcessTask {
    input: Vec<f32>,
    inner: Arc<Mutex<BiquadFilterInner>>,
}

#[napi]
impl Task for BiquadProcessTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut inner = self.inner.lock().unwrap();
        Ok(inner.process(&self.input))
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
impl BiquadFilter {
    #[napi(constructor)]
    pub fn new(coefficients: BiquadCoefficients) -> Self {
        BiquadFilter {
            inner: Arc::new(Mutex::new(BiquadFilterInner {
                coeffs: coefficients,
                x1: 0.0,
                x2: 0.0,
                y1: 0.0,
                y2: 0.0,
            })),
        }
    }

    #[napi]
    pub fn process(&self, input: Float32Array) -> AsyncTask<BiquadProcessTask> {
        AsyncTask::new(BiquadProcessTask {
            input: input.to_vec(),
            inner: self.inner.clone(),
        })
    }

    #[napi]
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.reset();
    }
}
