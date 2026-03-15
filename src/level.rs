use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi(object)]
pub struct LevelResult {
    pub rms: f64,
    pub rms_db: f64,
    pub peak: f64,
    pub peak_db: f64,
}

struct MeasureLevelTask {
    samples: Vec<f32>,
}

#[napi]
impl Task for MeasureLevelTask {
    type Output = LevelResult;
    type JsValue = LevelResult;

    fn compute(&mut self) -> Result<Self::Output> {
        if self.samples.is_empty() {
            return Ok(LevelResult {
                rms: 0.0,
                rms_db: f64::NEG_INFINITY,
                peak: 0.0,
                peak_db: f64::NEG_INFINITY,
            });
        }

        let mut sum_sq: f64 = 0.0;
        let mut peak: f32 = 0.0;

        for &s in &self.samples {
            let abs = s.abs();
            sum_sq += (s as f64) * (s as f64);
            if abs > peak {
                peak = abs;
            }
        }

        let rms = (sum_sq / self.samples.len() as f64).sqrt();
        let rms_db = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            f64::NEG_INFINITY
        };
        let peak_f64 = peak as f64;
        let peak_db = if peak_f64 > 0.0 {
            20.0 * peak_f64.log10()
        } else {
            f64::NEG_INFINITY
        };

        Ok(LevelResult {
            rms,
            rms_db,
            peak: peak_f64,
            peak_db,
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi]
pub fn measure_level(samples: Float32Array) -> AsyncTask<MeasureLevelTask> {
    AsyncTask::new(MeasureLevelTask {
        samples: samples.to_vec(),
    })
}
