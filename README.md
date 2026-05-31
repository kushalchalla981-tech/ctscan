# CT Reconstruction using LU Decomposition

Educational prototype demonstrating medical image reconstruction from CT projection data using LU factorization.

## Phase 1: Forward Model ✅

### Components
- `src/phantom.py`: Shepp-Logan phantom generation
- `src/projector.py`: Parallel-beam forward projector (builds sparse A and sinogram b)
- `validate_phase1.py`: Validation script with visualizations
- `tests/test_phase1.py`: Unit tests

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run validation
python validate_phase1.py

# Run tests
pytest tests/test_phase1.py -v
```

### Expected Output
- **Phantom**: 32×32 normalized image [0, 1]
- **System matrix A**: ~14,000 × 1,024 sparse matrix (>99% zeros)
- **Sinogram b**: ~14,000 measurements
- **Sparsity**: >95%
- **Forward model residual**: <1e-6

---

## Phase 2: LU Solver ✅

### Components
- `src/lud_solver.py`: Pivoted LU factorization with sparse/dense support
- `validate_phase2.py`: Solver validation on known systems
- `tests/test_phase2.py`: Unit tests

### Features
- Partial pivoting for numerical stability
- Automatic sparse/dense method selection
- Iterative refinement for accuracy improvement
- Condition number checking with warnings
- Fallback handling for ill-conditioned matrices

### Quick Start

```bash
# Run validation
python validate_phase2.py

# Run tests
pytest tests/test_phase2.py -v
```

### Expected Output
- Dense solver: residual <1e-15 for well-conditioned systems
- Sparse solver: SuperLU factorization for large systems
- Automatic method selection based on matrix size/type
- Warnings for ill-conditioned matrices (cond > 1e10)

---

## Phase 3: CT Reconstruction ✅

### Components
- `src/reconstructor.py`: Main reconstruction pipeline
- `src/metrics.py`: RMSE, PSNR, SSIM computation
- `validate_phase3.py`: Full pipeline validation
- `tests/test_phase3.py`: Unit tests

### Features
- LSQR least squares solver for overdetermined CT systems
- Iterative refinement option
- Comprehensive quality metrics
- Visualization: phantom, reconstruction, error map, sinogram

### Quick Start

```bash
# Run validation
python validate_phase3.py

# Run reconstruction
python src/reconstructor.py --size 32

# With iterative refinement
python src/reconstructor.py --size 32 --refine

# Run tests
pytest tests/test_phase3.py -v
```

### Results (32×32)
- **RMSE**: 0.0223 (< 0.05 ✓)
- **PSNR**: 33.38 dB (> 25 dB ✓)
- **SSIM**: 0.9947
- **Residual**: 2.58e-05 (< 1e-3 ✓)
- **With refinement**: RMSE 0.0012, PSNR 58.35 dB

---

## Phase 4: Noise Robustness ✅

### Components
- `src/noise.py`: Gaussian and Poisson noise addition utilities
- `validate_phase4.py`: Multi-level noise robustness validation
- `tests/test_phase4.py`: 12 unit tests

### Features
- Gaussian noise at configurable levels (0–20%)
- Poisson (photon) noise simulation
- Reproducible noise with seeded RNG
- Automatic metrics collection across noise levels
- Quality degradation curves (RMSE, PSNR, SSIM vs noise)
- Tikhonov regularization for noisy reconstructions

### Quick Start

```bash
# Run validation (generates degradation curves + visual comparison)
python validate_phase4.py

# Run tests
pytest tests/test_phase4.py -v
```

### Key Findings
1. **Unregularized LSQR catastrophically fails** at even 1% noise (RMSE jumps 0.02→2.56)
2. **Tikhonov regularization (damp=2.0)** enables graceful degradation
3. **Iterative refinement is harmful** with noisy data (overfits noise)

### Expected Results (regularized, damp=2.0)
| Noise | RMSE  | PSNR   | SSIM   |
|-------|-------|--------|--------|
| 0%    | 0.083 | 21.6 dB | 0.913 |
| 1%    | 0.095 | 20.4 dB | 0.890 |
| 5%    | 0.245 | 14.7 dB | 0.551 |
| 10%   | 0.467 | 12.1 dB | 0.252 |
| 20%   | 0.922 | 10.6 dB | 0.080 |

> Note: Regularization adds slight bias at 0% noise (RMSE 0.083 vs 0.022 unregularized)
> but is essential for any robustness to measurement noise.

### Outputs
- `phase4_degradation_curves.png`: RMSE/PSNR/SSIM vs noise level (unreg vs regularized)
- `phase4_noise_visual_comparison.png`: Reconstructions and error maps at 0/1/5/20% noise
