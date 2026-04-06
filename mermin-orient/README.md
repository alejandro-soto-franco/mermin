# mermin-orient

Multiscale structure tensor, k-atic order parameter fields, and nuclear ellipse fitting. Part of the [mermin](https://crates.io/crates/mermin) library.

## Features

- **Separable Gaussian blur**: zero-padded, truncated at 4 sigma
- **Scharr gradient operator**: better rotational symmetry than Sobel for orientation analysis
- **Structure tensor**: J_sigma(x) = G_sigma * (grad I tensor grad I) with eigendecomposition for local orientation theta(x) and coherence C(x)
- **Multiscale analysis**: structure tensor at logarithmically spaced scales (default: sigma = 1, 2, 4, 8, 16, 32 pixels), with optimal scale map
- **k-atic order parameter field**: psi_k(x, sigma) = C(x, sigma) * exp(ik * theta(x, sigma)) for arbitrary k, with per-region mean order computation
- **Nuclear ellipse fitting**: second central moments of DAPI nuclear masks give aspect ratio, orientation, and semi-axes without any model fitting

## Three Scales of Alignment

The multiscale structure tensor captures orientation at three physically distinct scales:

| Scale | sigma (pixels) | What it captures |
|-------|---------------|-----------------|
| Subcellular | 1-3 | Individual vimentin fiber bundles |
| Cellular | 8-16 | Whole-cell cytoskeletal alignment |
| Tissue | 32+ | Collective orientational order |

ROCK inhibition may destroy cellular-scale alignment while preserving subcellular fiber structure. The scale-space analysis reveals this.

## Usage

```rust
use mermin::orient::{structure_tensor, katic_order_field, fit_nuclear_ellipse};
use mermin::ImageField;

let st = structure_tensor(&vimentin_field, 4.0);
let psi2 = katic_order_field(&st, 2);

let ellipse = fit_nuclear_ellipse(&nuclear_pixels).unwrap();
println!("aspect ratio: {:.2}", ellipse.aspect_ratio);
```

## License

MIT
