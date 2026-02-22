use thiserror::Error;

#[derive(Error, Debug)]
pub enum BcmodelError {
    #[error("file too small to contain header size (need 8 bytes, got {0})")]
    FileTooSmall(usize),

    #[error("header size {header_size} exceeds available data {data_len}")]
    HeaderOverflow { header_size: u64, data_len: usize },

    #[error("invalid UTF-8 in JSON header: {0}")]
    InvalidUtf8(#[from] std::str::Utf8Error),

    #[error("invalid JSON header: {0}")]
    InvalidJson(#[from] serde_json::Error),

    #[error("tensor '{name}' offset [{begin}..{end}] exceeds binary section length {data_len}")]
    TensorOutOfBounds {
        name: String,
        begin: usize,
        end: usize,
        data_len: usize,
    },

    #[error("tensor '{name}' byte range size {size} is not aligned to 4 bytes (float32)")]
    TensorAlignment { name: String, size: usize },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
