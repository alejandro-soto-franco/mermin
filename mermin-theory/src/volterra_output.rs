// mermin-theory/src/volterra_output.rs

use crate::activity::estimate_activity;
use crate::frank::FrankEnergy;
use crate::landau_de_gennes::LdGParams;
use mermin_core::Real;
use serde::{Deserialize, Serialize};

/// Complete parameter set compatible with volterra's MarsParams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolterraParams {
    pub a_landau: Real,
    pub b_landau: Real,
    pub c_landau: Real,
    pub k_r: Real,
    pub zeta_eff: Real,
    pub frank_splay: Real,
    pub frank_bend: Real,
    pub frank_ratio: Real,
}

/// Build a volterra-compatible parameter set from mermin analysis results.
pub fn build_volterra_params(
    ldg: &LdGParams,
    frank: &FrankEnergy,
    n_defects: usize,
    image_area_um2: Real,
) -> VolterraParams {
    let zeta_eff = estimate_activity(n_defects, image_area_um2, ldg.k_elastic);

    VolterraParams {
        a_landau: ldg.a,
        b_landau: ldg.b,
        c_landau: ldg.c,
        k_r: ldg.k_elastic,
        zeta_eff,
        frank_splay: frank.splay,
        frank_bend: frank.bend,
        frank_ratio: frank.ratio,
    }
}

/// Serialize parameters to JSON string.
pub fn to_json(params: &VolterraParams) -> String {
    serde_json::to_string_pretty(params).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_json() {
        let params = VolterraParams {
            a_landau: -0.5,
            b_landau: 0.0,
            c_landau: 1.0,
            k_r: 0.01,
            zeta_eff: 0.001,
            frank_splay: 100.0,
            frank_bend: 80.0,
            frank_ratio: 1.25,
        };
        let json = to_json(&params);
        let parsed: VolterraParams = serde_json::from_str(&json).unwrap();
        assert!((parsed.a_landau - params.a_landau).abs() < 1e-15);
        assert!((parsed.frank_ratio - params.frank_ratio).abs() < 1e-15);
    }
}
