use napi::Status;

#[derive(Debug)]
pub struct DspError(pub String);

impl From<DspError> for napi::Error {
    fn from(e: DspError) -> Self {
        napi::Error::new(Status::GenericFailure, e.0)
    }
}

impl From<String> for DspError {
    fn from(s: String) -> Self {
        DspError(s)
    }
}

impl From<&str> for DspError {
    fn from(s: &str) -> Self {
        DspError(s.to_string())
    }
}
