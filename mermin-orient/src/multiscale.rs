// mermin-orient/src/multiscale.rs

use crate::structure_tensor::{structure_tensor, StructureTensorResult};
use mermin_core::{ImageField, Real};

/// Multiscale structure tensor analysis at logarithmically spaced scales.
pub struct MultiscaleResult {
    /// Structure tensor results at each scale, ordered by increasing sigma.
    pub scales: Vec<(Real, StructureTensorResult)>,
}

/// Compute structure tensor at multiple scales.
///
/// Default scales: [1, 2, 4, 8, 16, 32] pixels.
/// For each scale, produces theta(x, sigma) and coherence(x, sigma) fields.
pub fn multiscale_structure_tensor(
    image: &ImageField,
    sigmas: &[Real],
) -> MultiscaleResult {
    let scales = sigmas
        .iter()
        .map(|&sigma| (sigma, structure_tensor(image, sigma)))
        .collect();
    MultiscaleResult { scales }
}

/// For each pixel, find the scale sigma* that maximizes coherence.
/// Returns (optimal_sigma, max_coherence) fields.
pub fn optimal_scale_map(result: &MultiscaleResult) -> (ImageField, ImageField) {
    let (w, h) = if let Some((_, st)) = result.scales.first() {
        (st.theta.width, st.theta.height)
    } else {
        return (ImageField::zeros(0, 0), ImageField::zeros(0, 0));
    };

    let mut opt_sigma = ImageField::zeros(w, h);
    let mut max_coh = ImageField::zeros(w, h);

    for i in 0..w * h {
        let mut best_c = -1.0;
        let mut best_s = 0.0;
        for (sigma, st) in &result.scales {
            let c = st.coherence.data[i];
            if c > best_c {
                best_c = c;
                best_s = *sigma;
            }
        }
        opt_sigma.data[i] = best_s;
        max_coh.data[i] = best_c;
    }

    (opt_sigma, max_coh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiscale_produces_all_scales() {
        let field = ImageField::zeros(50, 50);
        let result = multiscale_structure_tensor(&field, &[1.0, 4.0, 16.0]);
        assert_eq!(result.scales.len(), 3);
    }
}
