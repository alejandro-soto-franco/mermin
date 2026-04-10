# mermin test data

This directory is gitignored. It contains collaborator-provided microscopy images for development and validation.

## Dataset: Human Vaginal Fibroblasts (hVFs)

All collaborator datasets are organized under `data/<collaborator>/<batch>/`.

## montano/postmeno-vestrogen-hVF-2026-04/

Source: Josh Montano, 2026-04-06 (initial batch), 2026-04-08 (full plate upload)

### Patient population
- Postmenopausal women on vaginal estrogen therapy

### Cell type
- Human vaginal fibroblasts, passage 4
- Isolated by explant method (tissue crawl-out in dish with fresh media over 2-3 weeks)

### Seeding
- 30,000 cells/well
- 96-well ibidi glass-bottom plate
- Surface area: 0.56 cm^2 per well

### Conditions (4 treatments x 3 replicates x 2 substrates)

Each treatment occupies 3 consecutive wells per row. Two rows per substrate provide biological replicates.

| Wells       | Treatment                                         |
|-------------|---------------------------------------------------|
| 01-03       | Control (untreated)                               |
| 04-06       | 17-beta-estradiol (20 nM)                         |
| 07-09       | hTGF-beta1 (2 ng/mL)                              |
| 10-12       | ROCKi pre-treatment (10 uM, 2h) + hTGF-beta1     |

### Substrate and row layout

| Row prefix | Substrate                          |
|------------|------------------------------------|
| `c`        | Untreated glass (row C)            |
| `d`        | Untreated glass (row D)            |
| `g`        | Collagen coat, 1h incubation (row G) |
| `h`        | Collagen coat, 1h incubation (row H) |

### Confluence
- Collagen-coated: 80-90% confluent
- Untreated glass: ~70% confluent
- NOTE: cell packing / nuclei count varies across conditions (less controlled in untreated glass)

### Imaging
- Fluorescence microscopy
- 16-bit unsigned integer (12-bit dynamic range, max ~4095)
- 4015 x 4015 pixels per frame

**Channel configurations:**

| Suffix | Channels | Frame 0        | Frame 1              | Frame 2              |
|--------|----------|----------------|----------------------|----------------------|
| `_dv`  | 2        | DAPI (470 nm)  | Vimentin (666 nm)    |                      |
| `_dsv` | 3        | DAPI (470 nm)  | SMAD (525 nm)        | Vimentin (666 nm)    |

- `_dv` = DAPI + Vimentin (standard 2-channel, most files)
- `_dsv` = DAPI + SMAD + Vimentin (3-channel, c08 and c12 only)
- File naming: `{row}{well}_{channels}.tif`

### Available files (49 total)

Place TIFFs in `data/montano/postmeno-vestrogen-hVF-2026-04/`.

**Glass (rows C and D, 24 files):**

| File          | Condition              | Channels | Notes |
|---------------|------------------------|----------|-------|
| c01-c03_dv    | Control                | 2 (dv)   |       |
| c04-c06_dv    | 17-beta-estradiol      | 2 (dv)   |       |
| c07_dv, c09_dv | hTGF-beta1            | 2 (dv)   |       |
| c08_dsv       | hTGF-beta1             | 3 (dsv)  | includes SMAD channel |
| c10-c11_dv    | ROCKi + hTGF-beta1     | 2 (dv)   |       |
| c12_dsv       | ROCKi + hTGF-beta1     | 3 (dsv)  | includes SMAD channel |
| d01-d03_dv    | Control                | 2 (dv)   |       |
| d04-d06_dv    | 17-beta-estradiol      | 2 (dv)   |       |
| d07-d09_dv    | hTGF-beta1             | 2 (dv)   |       |
| d10-d12_dv    | ROCKi + hTGF-beta1     | 2 (dv)   |       |

**Collagen (rows G and H, 24 files):**

| File          | Condition              | Channels | Notes |
|---------------|------------------------|----------|-------|
| g01-g03_dv    | Control                | 2 (dv)   |       |
| g04-g06_dv    | 17-beta-estradiol      | 2 (dv)   |       |
| g07-g09_dv    | hTGF-beta1             | 2 (dv)   | G09 normalised to g09 |
| g10-g12_dv    | ROCKi + hTGF-beta1     | 2 (dv)   |       |
| h01-h03_dv    | Control                | 2 (dv)   |       |
| h04-h06_dv    | 17-beta-estradiol      | 2 (dv)   |       |
| h07-h08_dv    | hTGF-beta1             | 2 (dv)   |       |
| h09_dv        | hTGF-beta1             | 2 (dv)   | from initial batch only (missing from 2026-04-08 upload) |
| h10-h12_dv    | ROCKi + hTGF-beta1     | 2 (dv)   |       |

**Calibration (1 file):**

| File | Description |
|------|-------------|
| jm008.3_ifg2vg3p_tgfbe2response_2hr_C05_sx_1_sy_1_w4.tif | Single-channel scale bar reference |

### Expected phenotypes
- TGF-beta1: myofibroblast activation, increased elongation and alignment (nematic order)
- ROCKi + TGF-beta1: more rounded nuclei, reduced alignment (ROCK inhibition disrupts actomyosin contractility)
- 17-beta-estradiol: modulatory effect on TGF-beta/Smad signaling
- Collagen coat: generally improves adhesion and alignment vs bare glass

### Reference
- Varelas et al., "TAZ controls Smad nucleocytoplasmic shuttling and regulates human embryonic stem-cell self-renewal", Nature Cell Biology 10(7):837-48, 2008. PMID: 18568018
  - Context for TGF-beta/Smad/TAZ signaling pathway, not imaging methodology

### Imaging system
- Molecular Devices ImageXpress Pico ("Pico Sirius Max")
- Objective: 10x Leica Fluotar air (NA 0.32)
- Camera: Sony CMOS 5 MP (likely IMX264, 2448 x 2048, 3.45 um pixel pitch)
- Images are 2x2 stitched montages (4015 x 4015 px)

### Pixel size
- Estimated: 0.345 um/pixel (3.45 um sensor pitch / 10x magnification)
- Total FOV per image: ~1.39 x 1.39 mm
- Pending confirmation from CellReporterXpress acquisition metadata

### Calibration image details

The file `jm008.3_ifg2vg3p_tgfbe2response_2hr_C05_sx_1_sy_1_w4.tif` serves as the pixel-size calibration reference.

**Filename fields:**
- `jm008.3`: Josh Montano sample 008, subfield 3
- `ifg2vg3p`: antibody/staining shorthand
- `tgfbe2response_2hr`: TGF-beta2 response, 2-hour treatment
- `C05`: well C05 (96-well plate)
- `sx_1_sy_1`: stage/site position (1,1) in montage grid
- `w4`: wavelength/channel 4 (666 nm, vimentin/Cy5)

**Properties:**
- Single channel, 4015 x 4015 px, uint16
- Acquired with MetaMorph on 2026-04-01 at 10X (NA 0.5)
- Embedded MetaMorph spatial calibration: 0.69 um/px
- Contains a 600 um scale bar (870 px), used to determine pixel size for the numerical-tests report: 0.6897 +/- 0.001 um/px
- From a separate experiment (TGF-beta2, well C05) with different well prefix convention

### Collaborator
- Josh Montano (provided dataset, experimental design, and imaging metadata)
