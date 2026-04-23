//! Gradient descent utilities

/// Compute gradient of loss.
pub fn compute_gradient(loss: &[f32], params: &[f32]) -> Vec<f32> {
    // Stub: return zeros
    vec![0.0f32; params.len()]
}

/// Update parameters with gradient.
pub fn update_params(params: &mut [f32], gradient: &[f32], lr: f32) {
    for (p, g) in params.iter_mut().zip(gradient.iter()) {
        *p -= lr * g;
    }
}
