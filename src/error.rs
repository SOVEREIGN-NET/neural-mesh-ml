//! Error types for neural mesh operations

use thiserror::Error;

#[derive(Debug, Error)]
pub enum NeuralMeshError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("Inference error: {0}")]
    InferenceFailed(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Timeout after {0}ms")]
    Timeout(u64),

    #[error("Training error: {0}")]
    TrainingFailed(String),

    #[error("Synchronization error: {0}")]
    SyncFailed(String),
}

pub type Result<T> = std::result::Result<T, NeuralMeshError>;
