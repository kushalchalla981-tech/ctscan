# CT Reconstruction using LU Decomposition

Educational prototype demonstrating medical image reconstruction from CT projection data using LU factorization and least-squares solvers.

## Quick Start

```bash
pip install -r requirements.txt
python main.py reconstruct                       # Shepp-Logan phantom
python main.py reconstruct --refine              # with iterative refinement
python main.py upload                            # file dialog → your image
python main.py interactive                       # menu-driven mode
python main.py info
```

## CLI Reference

| Command | Description |
|---|---|
| `python main.py reconstruct [options]` | Run full reconstruction pipeline |
| `python main.py upload [options]` | Pick an image → simulate CT reconstruction |
| `python main.py validate [--phase N] [--all]` | Run validation checks |
| `python main.py noise [options]` | Noise robustness sweep |
| `python main.py interactive` | Menu-driven interactive mode |
| `python main.py info` | Project information |

### `reconstruct`

```bash
# Synthetic phantom
python main.py reconstruct --size 32 --refine --output result.png
# Real DICOM scan
python main.py reconstruct --input samples/CT-brain.dcm --compare --save-metrics results.json
```

| Flag | Default | Description |
|---|---|---|
| `--size` | 32 | Phantom dimension (pixels) |
| `--refine` | off | Apply iterative refinement |
| `--method` | auto | Solver: auto, sparse, or dense |
| `--input` / `-i` | none | Path to DICOM or image file |
| `--compare` / `-c` | none | Save 4-panel comparison plot |
| `--save-metrics` / `-m` | none | Export metrics to JSON |
| `--output` / `-o` | none | Save final plot to file |

### `validate`

```bash
python main.py validate --all
python main.py validate --phase 2
```

| Flag | Description |
|---|---|
| `--phase` 1–4 | Run a specific validation |
| `--all` | Run all validations |
| `--size` | Phantom size (default: 32) |

### `noise`

```bash
python main.py noise --levels 0 1 5 10 20 --regularize --plot noise.png
```

| Flag | Default | Description |
|---|---|---|
| `--levels` | 0 1 5 10 20 | Noise levels in percent |
| `--regularize` | off | Enable Tikhonov regularization |
| `--size` | 32 | Phantom size |
| `--plot` / `-p` | none | Save visual comparison |

### `upload`

Opens a file dialog (or accepts `--file`) to pick any image. The image is
used as the "ground truth" — we simulate CT X-ray projections through it,
then reconstruct the original from those projections.

```bash
python main.py upload                          # file dialog pops up
python main.py upload --file my_photo.png      # skip dialog, go direct
python main.py upload -f my_photo.png --compare --size 48
```

| Flag | Default | Description |
|---|---|---|
| `--file` / `-f` | none | Direct path to image (no dialog) |
| `--size` | 32 | Phantom dimension (pixels) |
| `--refine` | off | Apply iterative refinement |
| `--method` | auto | Solver: auto, sparse, or dense |
| `--compare` / `-c` | none | Save 4-panel comparison plot |
| `--save-metrics` / `-m` | none | Export metrics to JSON |

### `interactive`

Launches a numbered menu with options for phantom reconstruction, image upload, validations, noise sweep, and project info — no flags needed.

## Samples

| File | Type | Description |
|---|---|---|
| `CT-brain.dcm` | Real CT (525 KB) | Head scan — Barre's Collection |
| `CT-chest.dcm` | Real CT (145 KB) | Chest scan — Barre's Collection |
| `CT-ankle.dcm` | Real CT (525 KB) | Ankle scan — Barre's Collection |
| `CT-small.dcm` | Real CT (39 KB) | 128×128 CT — pydicom test fixture |
| `phantom-shepp-logan.dcm` | Synthetic DICOM | Shepp-Logan phantom (128×128) |
| `phantom-rings.dcm` | Synthetic DICOM | Concentric rings test pattern |
| `phantom-resolution.dcm` | Synthetic DICOM | Resolution bar pattern |
| `phantom-gradient.dcm` | Synthetic DICOM | Intensity gradient |
| `checkerboard.png` | Image (118 B) | 128×128 checkerboard pattern |
| `rings.png` | Image (1 KB) | Concentric rings |
| `gradient.png` | Image (192 B) | Linear intensity gradient |
| `shapes.png` | Image (392 B) | Geometric shapes |

> Synthetic DICOM files are valid CT DICOMs generated with pydicom.
> PNG files can be used with `--input` without any DICOM library.

## Architecture

```
main.py                  ← Single entry point (argparse + rich)
src/
├── loader.py            DICOM/raster image loader
├── phantom.py           Shepp-Logan phantom generator
├── projector.py         Parallel-beam ray tracer → sparse A + sinogram b
├── lud_solver.py        LU decomposition with pivoting + LSQR
├── metrics.py           RMSE, PSNR, SSIM
├── reconstructor.py     Full reconstruction pipeline
├── noise.py             Gaussian/Poisson noise + robustness testing
└── validate.py          Consolidated validation suite
samples/                 12 example files (real DICOM + synthetic + PNG)
tests/                   31 unit tests
demo.ipynb               Jupyter notebook walkthrough
COLLEGE_REPORT.md        Full project report
```

## Results (32×32, clean)

| Metric | Basic | With refinement |
|---|---|---|
| RMSE | 0.0223 | 0.0012 |
| PSNR | 33.38 dB | 58.35 dB |
| SSIM | 0.995 | 1.000 |
| Residual | 2.6e-05 | 2.3e-07 |

## Noise Robustness (regularized, damp=2.0)

| Noise | RMSE | PSNR | SSIM |
|---|---|---|---|
| 0% | 0.083 | 21.6 dB | 0.913 |
| 5% | 0.245 | 14.7 dB | 0.551 |
| 10% | 0.467 | 12.1 dB | 0.252 |
| 20% | 0.922 | 10.6 dB | 0.080 |

> Regularization adds slight bias at 0% noise (RMSE 0.083 vs 0.022) but is essential for robustness. Unregularized LSQR catastrophically fails at even 1% noise.

## Testing

```bash
pytest tests/ -v
```

## References

- Shepp & Logan (1974). "The Fourier reconstruction of a head section."
- Paige & Saunders (1982). "LSQR: An algorithm for sparse linear equations."
- Barre's DICOM Collection: https://barre.dev/medical/samples/
