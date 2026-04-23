//! Isolation Forest for anomaly detection

/// Isolation Forest model.
pub struct IsolationForest {
    _private: (),
}

impl IsolationForest {
    pub fn new(n_trees: usize, sample_size: usize) -> Self {
        IsolationForest { _private: () }
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) {
        // Stub
    }

    pub fn score(&self, point: &[f32]) -> f32 {
        0.5 // Stub: middle score
    }
}
