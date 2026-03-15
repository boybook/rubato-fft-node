use napi::bindgen_prelude::*;
use napi_derive::napi;

// --- Int16 <-> Float32 ---

struct Int16ToFloat32Task {
    input: Vec<i16>,
}

#[napi]
impl Task for Int16ToFloat32Task {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.input.iter().map(|&s| s as f32 / 32768.0).collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
pub fn int16_to_float32(input: Int16Array) -> AsyncTask<Int16ToFloat32Task> {
    AsyncTask::new(Int16ToFloat32Task {
        input: input.to_vec(),
    })
}

struct Float32ToInt16Task {
    input: Vec<f32>,
}

#[napi]
impl Task for Float32ToInt16Task {
    type Output = Vec<i16>;
    type JsValue = Int16Array;

    fn compute(&mut self) -> Result<Self::Output> {
        Ok(self.input.iter().map(|&s| {
            let clamped = s.clamp(-1.0, 1.0);
            (clamped * 32767.0) as i16
        }).collect())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Int16Array::new(output))
    }
}

#[napi]
pub fn float32_to_int16(input: Float32Array) -> AsyncTask<Float32ToInt16Task> {
    AsyncTask::new(Float32ToInt16Task {
        input: input.to_vec(),
    })
}

// --- Interleave / Deinterleave ---

struct InterleaveTask {
    channels: Vec<Vec<f32>>,
}

#[napi]
impl Task for InterleaveTask {
    type Output = Vec<f32>;
    type JsValue = Float32Array;

    fn compute(&mut self) -> Result<Self::Output> {
        if self.channels.is_empty() {
            return Ok(vec![]);
        }
        let num_channels = self.channels.len();
        let num_frames = self.channels[0].len();
        let mut output = Vec::with_capacity(num_channels * num_frames);
        for frame in 0..num_frames {
            for ch in 0..num_channels {
                output.push(if frame < self.channels[ch].len() {
                    self.channels[ch][frame]
                } else {
                    0.0
                });
            }
        }
        Ok(output)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(Float32Array::new(output))
    }
}

#[napi]
pub fn interleave(channels: Vec<Float32Array>) -> AsyncTask<InterleaveTask> {
    AsyncTask::new(InterleaveTask {
        channels: channels.iter().map(|c| c.to_vec()).collect(),
    })
}

struct DeinterleaveTask {
    interleaved: Vec<f32>,
    num_channels: u32,
}

#[napi]
impl Task for DeinterleaveTask {
    type Output = Vec<Vec<f32>>;
    type JsValue = Vec<Float32Array>;

    fn compute(&mut self) -> Result<Self::Output> {
        let nc = self.num_channels as usize;
        if nc == 0 {
            return Err(napi::Error::from_reason("numChannels must be > 0"));
        }
        let num_frames = self.interleaved.len() / nc;
        let mut channels: Vec<Vec<f32>> = (0..nc).map(|_| Vec::with_capacity(num_frames)).collect();
        for (i, &sample) in self.interleaved.iter().enumerate() {
            channels[i % nc].push(sample);
        }
        Ok(channels)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into_iter().map(Float32Array::new).collect())
    }
}

#[napi]
pub fn deinterleave(
    interleaved: Float32Array,
    num_channels: u32,
) -> AsyncTask<DeinterleaveTask> {
    AsyncTask::new(DeinterleaveTask {
        interleaved: interleaved.to_vec(),
        num_channels,
    })
}
