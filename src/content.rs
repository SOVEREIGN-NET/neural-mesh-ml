//! Content analysis and compression feedback

use serde::{Deserialize, Serialize};

/// Type of content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Binary,
    Structured,
    Compressed,
}

/// Profile of content characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentProfile {
    pub content_type: ContentType,
    pub entropy: f32,
    pub redundancy: f32,
    pub size_bytes: usize,
}

/// Feedback for compression optimization.
pub trait CompressionFeedback {
    fn ratio(&self) -> f32;
    fn latency_ms(&self) -> u64;
    fn cpu_cost(&self) -> f32;
}
