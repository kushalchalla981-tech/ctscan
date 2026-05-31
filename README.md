# CT Reconstruction using LU Decomposition

Educational prototype demonstrating medical image reconstruction from CT projection data using LU factorization and least-squares solvers.

## Quick Start

```bash
pip install -r requirements.txt
python main.py reconstruct          # 32×32 reconstruction
python main.py reconstruct --refine # with iterative refinement
python main.py info                 # system info
python main.py interactive          # menu-driven TUI
```

## CLI Reference

| Command | Description |
|---|---|
| `python main.py reconstruct [options]` | Run full reconstruction pipeline |
| `python main.py validate [--phase N] [--all]` | Run validation checks |
| `python main.py noise [options]` | Noise robustness sweep |
| `python main.py interactive` | Menu-driven interactive mode |
| `python main.py info` | Project information |

### `reconstruct`

```
python main.py reconstruct --size 32 --refine --method auto --output result.png
```

| Flag | Default | Description |
|---|---|---|
| `--size` | 32 | Phantom dimension (pixels) |
| `--refine` | off | Apply iterative refinement |
| `--method` | auto | Solver: auto, sparse, or dense |
| `--output` / `-o` | none | Save visualization to file |

### `validate`

```
python main.py validate --all
python main.py validate --phase 2
```

| Flag | Description |
|---|---|
| `--phase` 1–4 | Run a specific validation |
| `--all` | Run all validations |
| `--size` | Phantom size (default: 32) |

### `noise`

```
python main.py noise --levels 0 1 5 10 20 --regularize --plot noise.png
```

| Flag | Default | Description |
|---|---|---|
| `--levels` | 0 1 5 10 20 | Noise levels in percent |
| `--regularize` | off | Enable Tikhonov regularization |
| `--size` | 32 | Phantom size |
| `--plot` / `-p` | none | Save visual comparison |

### `interactive`

Launches a numbered menu for quick access to all pipelines without CLI flags.

## Architecture

```
main.py                  ← Single entry point (argparse + rich)
src/
├── phantom.py           Shepp-Logan phantom generator
├── projector.py         Parallel-beam ray tracer → sparse A + sinogram b
├── lud_solver.py        LU decomposition with pivoting + LSQR
├── metrics.py           RMSE, PSNR, SSIM
├── reconstructor.py     Full reconstruction pipeline
├── noise.py             Gaussian/Poisson noise + robustness testing
└── validate.py          Consolidated validation suite
tests/
├── test_phase1.py       6 tests — forward model
├── test_phase2.py       7 tests — LU solver
├── test_phase3.py       6 tests — reconstruction
└── test_phase4.py       12 tests — noise robustness
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

> Regularization adds slight bias at 0% noise but is essential for robustness.

## Testing

```bash
pytest tests/ -v
```
