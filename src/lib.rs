//! Neural Mesh ML Library
//!
//! ML/AI optimization layer for routing, compression, prefetching, and anomaly detection.
//!
//! # Features
//!
//! - **Reinforcement Learning** (PPO) for routing decisions
//! - **LSTM** for predictive prefetching
//! - **Isolation Forest** for Byzantine anomaly detection
//! - **Federated Learning** (FedAvg) for distributed training
//! - **ONNX Inference** for model deployment
//!
//! # Example
//!
//! ```rust
//! use lib_neural_mesh::{RlRouter, AnomalySentry, PredictivePrefetcher};
//!
//! // RL-based routing
//! let router = RlRouter::new();
//! let action = router.decide(&network_state)?;
//!
//! // Anomaly detection
//! let sentry = AnomalySentry::new();
//! let report = sentry.analyze(&node_metrics)?;
//! ```

// Core modules
pub mod error;
pub mod inference;
pub mod router;
pub mod compressor;
pub mod prefetch;
pub mod anomaly;
pub mod content;
pub mod codec_learner;
pub mod distributed;
pub mod parallel_shard_stream;
pub mod semantic_channeling;

// ML implementations
pub mod ml;

// Re-exports
pub use error::{NeuralMeshError, Result};
pub use inference::InferenceEngine;
pub use router::{RlRouter, NetworkState, RoutingAction};
pub use compressor::{NeuroCompressor, ContentEmbedder, Embedding};
pub use prefetch::{PredictivePrefetcher, AccessPattern, PredictionResult};
pub use anomaly::{AnomalySentry, NodeMetrics, AnomalyReport, AnomalySeverity, ThreatType};
pub use content::{ContentType, ContentProfile, CompressionFeedback};
pub use codec_learner::{AdaptiveCodecLearner, CodecLearnerConfig, LearnedCodecParams};
pub use distributed::{DistributedTrainingCoordinator, FedAvgResult, ModelSyncMessage};

// Constants
pub const PROTOCOL_VERSION: u8 = 1;
pub const DEFAULT_INFERENCE_TIMEOUT_MS: u64 = 50;
pub const MAX_MODEL_SIZE: usize = 100 * 1024 * 1024; // 100 MB
