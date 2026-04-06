# mermin test data

This directory is gitignored. It contains collaborator-provided microscopy images for development and validation.

## Dataset: Human Vaginal Fibroblasts (hVFs)

Source: collaborator-provided, 2026-04-06

### Cell type
- Human vaginal fibroblasts, passage 4
- Isolated by explant method (tissue crawl-out in dish with fresh media over 2-3 weeks)

### Seeding
- 30,000 cells/well
- 96-well ibidi glass-bottom plate
- Surface area: 0.56 cm^2 per well

### Conditions (4 treatments x 3 replicates)

| Wells   | Treatment                                         |
|---------|---------------------------------------------------|
| d01-d03 | Control (untreated)                               |
| d04-d06 | 17-beta-estradiol (20 nM)                         |
| d07-d09 | hTGF-beta1 (2 ng/mL)                              |
| d10-d12 | ROCKi pre-treatment (10 uM, 2h) + hTGF-beta1     |

### Substrate
- `d` prefix = no collagen coating (untreated glass)
- `h` prefix = collagen coat (1h incubation)

### Confluence
- Collagen-coated: 80-90% confluent
- Untreated glass: ~70% confluent
- NOTE: cell packing / nuclei count varies across conditions (less controlled in untreated glass)

### Imaging
- Fluorescence microscopy
- 2 channels per TIFF (multi-frame):
  - Frame 0: DAPI (nuclei)
  - Frame 1: Vimentin (intermediate filaments)
- 16-bit unsigned integer (12-bit dynamic range, max ~4095)
- 4015 x 4015 pixels per frame
- File naming: `{substrate}{well}_dv.tif`

### Available files (8 of 24 total, 1 replicate per condition x 2 substrates)

| File        | Condition              | Substrate |
|-------------|------------------------|-----------|
| d03_dv.tif  | Control                | Glass     |
| d05_dv.tif  | 17-beta-estradiol      | Glass     |
| d09_dv.tif  | hTGF-beta1             | Glass     |
| d12_dv.tif  | ROCKi + hTGF-beta1     | Glass     |
| h02_dv.tif  | Control                | Collagen  |
| h05_dv.tif  | 17-beta-estradiol      | Collagen  |
| h09_dv.tif  | hTGF-beta1             | Collagen  |
| h12_dv.tif  | ROCKi + hTGF-beta1     | Collagen  |

### Expected phenotypes
- TGF-beta1: myofibroblast activation, increased elongation and alignment (nematic order)
- ROCKi + TGF-beta1: more rounded nuclei, reduced alignment (ROCK inhibition disrupts actomyosin contractility)
- 17-beta-estradiol: modulatory effect on TGF-beta/Smad signaling
- Collagen coat: generally improves adhesion and alignment vs bare glass

### Reference
- Varelas et al., "TAZ controls Smad nucleocytoplasmic shuttling and regulates human embryonic stem-cell self-renewal", Nature Cell Biology 10(7):837-48, 2008. PMID: 18568018
  - Context for TGF-beta/Smad/TAZ signaling pathway, not imaging methodology

### Pixel size
- Not yet confirmed from microscope metadata. Estimate needed from collaborator.
- ibidi 96-well glass-bottom plates typically imaged at 10x or 20x.
