// mermin-stats/src/permutation.rs

use mermin_core::Real;
use rand::prelude::*;

/// Result of a permutation test.
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// Observed test statistic.
    pub observed: Real,
    /// Two-sided p-value.
    pub p_value: Real,
    /// Number of permutations.
    pub n_permutations: usize,
}

/// Two-sample permutation test for comparing a statistic between two conditions.
///
/// `values_a`: measurements from condition A.
/// `values_b`: measurements from condition B.
/// `statistic`: function that computes the test statistic (e.g., difference of means).
///   Takes two slices (a, b) and returns the statistic.
/// `n_permutations`: number of random permutations.
/// `seed`: random seed.
pub fn permutation_test<F>(
    values_a: &[Real],
    values_b: &[Real],
    statistic: F,
    n_permutations: usize,
    seed: u64,
) -> PermutationTestResult
where
    F: Fn(&[Real], &[Real]) -> Real,
{
    let observed = statistic(values_a, values_b);
    let na = values_a.len();

    // Pool all values
    let mut pooled: Vec<Real> = Vec::with_capacity(na + values_b.len());
    pooled.extend_from_slice(values_a);
    pooled.extend_from_slice(values_b);

    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_extreme = 0usize;

    for _ in 0..n_permutations {
        pooled.shuffle(&mut rng);
        let perm_stat = statistic(&pooled[..na], &pooled[na..]);
        if perm_stat.abs() >= observed.abs() {
            n_extreme += 1;
        }
    }

    let p_value = (n_extreme + 1) as Real / (n_permutations + 1) as Real;

    PermutationTestResult {
        observed,
        p_value,
        n_permutations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_high_p() {
        let a: Vec<Real> = (0..50).map(|i| i as Real).collect();
        let b: Vec<Real> = (0..50).map(|i| i as Real).collect();

        let result = permutation_test(
            &a,
            &b,
            |a, b| {
                let ma: Real = a.iter().sum::<Real>() / a.len() as Real;
                let mb: Real = b.iter().sum::<Real>() / b.len() as Real;
                ma - mb
            },
            999,
            42,
        );
        assert!(
            result.p_value > 0.05,
            "identical distributions should have high p-value, got {}",
            result.p_value
        );
    }

    #[test]
    fn different_distributions_low_p() {
        let a: Vec<Real> = vec![0.0; 50];
        let b: Vec<Real> = vec![100.0; 50];

        let result = permutation_test(
            &a,
            &b,
            |a, b| {
                let ma: Real = a.iter().sum::<Real>() / a.len() as Real;
                let mb: Real = b.iter().sum::<Real>() / b.len() as Real;
                ma - mb
            },
            999,
            42,
        );
        assert!(
            result.p_value < 0.01,
            "very different distributions should have low p-value, got {}",
            result.p_value
        );
    }
}
