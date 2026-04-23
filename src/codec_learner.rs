//! Adaptive codec parameter learning

use serde::{Deserialize, Serialize};
use crate::Result;

/// Configuration for codec learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecLearnerConfig {
    pub learning_rate: f32,
    pub exploration_rate: f32,
}

impl Default for CodecLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            exploration_rate: 0.1,
        }
    }
}

/// Learned codec parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedCodecParams {
    pub use_bwt: bool,
    pub use_mtf: bool,
    pub range_coder_order: usize,
}

/// Adaptive codec learner using RL.
pub struct AdaptiveCodecLearner {
    config: CodecLearnerConfig,
}

impl AdaptiveCodecLearner {
    pub fn new(config: CodecLearnerConfig) -> Self {
        Self { config }
    }

    /// Learn optimal codec parameters from feedback.
    pub fn learn(&self, feedback: &dyn crate::content::CompressionFeedback) -> Result<LearnedCodecParams> {
        // Stub: return default params
        Ok(LearnedCodecParams {
            use_bwt: true,
            use_mtf: true,
            range_coder_order: 1,
        })
    }
}
