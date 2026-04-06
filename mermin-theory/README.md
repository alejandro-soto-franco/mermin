# mermin-theory

Continuum theory parameter extraction: Frank elastic energy, Landau-de Gennes fitting, and activity estimation. Part of the [mermin](https://crates.io/crates/mermin) library.

## Features

- **Frank energy decomposition**: compute splay (div n)^2 and bend (n x curl n)^2 from a 2D director field on a regular grid using central differences. The ratio F_splay/F_bend distinguishes contractile-like (splay-dominated) from extensile-like (bend-dominated) active behaviour.
- **Landau-de Gennes parameter fitting**: estimate (a, b, c, K) from the experimental Q-tensor field via moment matching on the scalar order parameter distribution and correlation length
- **Activity estimation**: zeta_eff from defect density via mean-field active nematic scaling (defect spacing ~ sqrt(K/zeta))
- **Volterra-compatible output**: fitted parameters serialised as JSON matching [volterra](https://crates.io/crates/volterra-nematic)'s `MarsParams` field names for forward simulation

## Connecting Experiment to Theory

mermin-theory closes the loop from measurement to model. The extracted parameters (a, b, c, K, zeta_eff) fully specify a Beris-Edwards active nematic model. Future work: feed these into volterra for deterministic simulation and pathwise for stochastic ensemble forecasting.

## Usage

```rust
use mermin::theory::{frank_energy, estimate_ldg_params, build_volterra_params, to_json};

let frank = frank_energy(&thetas, nx, ny, pixel_size_um);
println!("splay/bend ratio: {:.2}", frank.ratio);

let ldg = estimate_ldg_params(&s_values, correlation_length, pixel_size_um);
let params = build_volterra_params(&ldg, &frank, n_defects, image_area_um2);
std::fs::write("params.json", to_json(&params))?;
```

## License

MIT
