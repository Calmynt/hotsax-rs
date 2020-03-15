use num::Float;
use std::ops::Deref;

#[inline]
/// Returns the mean of the list
pub fn mean<R, N>(data: &R) -> N where R: Deref<Target=[N]>, N: Float {
    data.iter().fold(N::zero(), |a, b| a+*b) / N::from(data.as_ref().len()).unwrap()
}

/// Returns the standard deviation of the list
pub fn std_dev<R, N>(data: &R) -> N where R: Deref<Target=[N]>, N: Float {
    let mean = mean(data);

    let sum = data.as_ref()
        .iter()
        .map(|e| (*e- N::from(mean).unwrap()).powi(2))
        .fold(N::zero(), |a, b| a+b);

    (sum/ N::from(data.len()).unwrap()).sqrt()
}

/// Computes the z-normalisation of the list itself.
pub fn znorm<R, N>(data: &R) -> Vec<N> where R: Deref<Target=[N]>, N: Float {
    let mean = mean(data);
    let std_dev = std_dev(data);
    data.iter().map(|e| (*e-mean) / std_dev).collect()
}

/// Calculates the gaussian distance between two lists of floats.
pub fn gaussian<N>(q: &[N], c: &[N]) -> N where N: Float {
    let sum = q
        .iter()
        .zip(c)
        .map(|(qi, ci)| (*qi - *ci).powi(2))
        .fold(N::zero(), |acc, x| acc + x);

    sum.sqrt()
}