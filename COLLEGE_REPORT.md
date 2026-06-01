# CT Reconstruction using LU Decomposition

## College Project Report

---

## Abstract

This project implements a computed tomography (CT) reconstruction pipeline from first principles using linear algebra. Starting from a known phantom (or real DICOM scan), we simulate X-ray projections to build a sparse system matrix **A** and measurement vector **b**. We then solve the linear system \(Ax = b\) using LSQR, an iterative least-squares solver. The pipeline includes pivoted LU decomposition for square subsystems, Tikhonov regularization for noise robustness, and comprehensive quality metrics (RMSE, PSNR, SSIM). The system achieves a reconstruction RMSE of 0.022 (PSNR 33.4 dB, SSIM 0.995) on a 32x32 phantom, improving to RMSE 0.001 with iterative refinement.

---

## 1. Introduction

### 1.1 The CT Inverse Problem

In X-ray CT, a scanner measures the attenuation of X-rays passing through a patient from multiple angles. The goal of reconstruction is to recover the spatial distribution of attenuation coefficients — i.e., the image itself.

Mathematically, this is a linear inverse problem:

\[
A x = b
\]

Where:
- \(x\) is the unknown image (flattened into a vector of pixel intensities)
- \(A\) is the system matrix where each row encodes one ray's path through the pixels
- \(b\) is the sinogram — the measured projection data

### 1.2 Why This is Hard

- The system is **overdetermined**: ~1,400 equations for ~1,000 unknowns
- The matrix **A** is **highly sparse** (~91% zeros) and **ill-conditioned**
- The naive approach (forming \(A^T A\) and solving) squares the condition number, making the problem worse
- Real measurements contain **noise**, which unregularized solvers amplify catastrophically

---

## 2. Methodology

### 2.1 Forward Model

**Phantom Generation** (`src/phantom.py`):
A modified Shepp-Logan phantom is generated from 10 ellipses with varying intensities, normalized to [0, 1]. This serves as the ground truth.

**Ray Tracing** (`src/projector.py`):
For each angle and detector position, a ray is traced through the pixel grid. The intersection length of each ray with each pixel becomes the entry in **A**. With \(n\) angles and \(d\) detectors, we get \(n \times d\) measurements.

For a 32x32 phantom:
- Angles: 32 (0° to 180°, evenly spaced)
- Detectors: 44 (~32 × 1.4)
- Matrix **A**: 1,408 × 1,024 with ~127,000 non-zeros (91% sparse)
- Memory: ~0.5 MB

### 2.2 LU Decomposition (`src/lud_solver.py`)

For square systems, we implement **PA = LU** factorization with partial pivoting:

```
For each column k:
  1. Find pivot: max |A[k:n, k]|
  2. Swap rows (partial pivoting for stability)
  3. Eliminate below the pivot
```

**Forward substitution**: solve \(Ly = Pb\)
**Backward substitution**: solve \(Ux = y\)

For **rectangular (overdetermined)** systems, we use SciPy's LSQR solver — an iterative method equivalent to conjugate gradient on the normal equations, but without forming \(A^T A\) explicitly.

### 2.3 Iterative Refinement

Given an initial solution \(x_0\), we iterate:

1. Compute residual \(r = b - A x_k\)
2. Solve \(A \Delta x = r\) for the correction
3. Update \(x_{k+1} = x_k + \Delta x\)

This improves accuracy for clean data but **overfits noise** when measurements are noisy.

### 2.4 Tikhonov Regularization

For noisy measurements, we solve the damped least-squares problem:

\[
\min_x \|Ax - b\|^2 + \lambda \|x\|^2
\]

This penalizes large solution components, suppressing noise amplification. The LSQR solver supports this natively via the `damp` parameter (\(damp = \sqrt{\lambda}\)).

### 2.5 Quality Metrics (`src/metrics.py`)

- **RMSE** (Root Mean Square Error): \(\sqrt{\frac{1}{n}\sum(x_{true} - x_{rec})^2}\)
- **PSNR** (Peak Signal-to-Noise Ratio): \(20 \log_{10}(\frac{max}{\text{RMSE}})\)
- **SSIM** (Structural Similarity): compares luminance, contrast, and structure
- **Relative Error**: \(\|x_{true} - x_{rec}\| / \|x_{true}\|\)

---

## 3. Results

### 3.1 Clean Reconstruction (32×32)

| Metric | Basic LSQR | With Refinement |
|---|---|---|
| RMSE | 0.0223 | **0.0012** |
| PSNR | 33.38 dB | **58.35 dB** |
| SSIM | 0.9947 | **1.0000** |
| Residual | 2.58 × 10⁻⁵ | 2.34 × 10⁻⁷ |

All success criteria met:
- RMSE < 0.05: **0.022** ✓
- PSNR > 25 dB: **33.4 dB** ✓
- Residual < 10⁻³: **2.6 × 10⁻⁵** ✓

### 3.2 Noise Robustness

Without regularization, even 1% Gaussian noise causes catastrophic failure (RMSE jumps from 0.022 to 2.56). With Tikhonov regularization (damp=2.0), the system degrades gracefully:

| Noise | RMSE | PSNR | SSIM | Visual Quality |
|---|---|---|---|---|
| 0% | 0.083 | 21.6 dB | 0.913 | Slight bias from regularization |
| 1% | 0.095 | 20.4 dB | 0.890 | Nearly identical to clean |
| 5% | 0.245 | 14.7 dB | 0.551 | Structure recognizable |
| 10% | 0.467 | 12.1 dB | 0.252 | Major features visible |
| 20% | 0.922 | 10.6 dB | 0.080 | Heavily degraded |

**Key finding**: Iterative refinement is **harmful** with noisy data — it reduces the residual by fitting noise, which amplifies artifacts in the reconstruction.

### 3.3 Real DICOM Reconstruction

The pipeline was tested on three public-domain CT DICOM samples downloaded from [Barre's Collection](https://barre.dev/medical/samples/):

| Sample | Anatomy | Source |
|---|---|---|
| CT-brain.dcm | Head | CT-MONO2-16-brain |
| CT-chest.dcm | Chest | CT-MONO2-16-chest |
| CT-ankle.dcm | Ankle | CT-MONO2-16-ankle |

Each sample is loaded as the ground truth, then simulated projections are generated and reconstructed. Quality metrics are computed against the original image.

---

## 4. Discussion

### 4.1 Why LU Decomposition?

LU decomposition is a foundational linear algebra technique taught in every numerical methods course. This project demonstrates that real-world problems often require more than textbook algorithms — the CT system is rectangular, ill-conditioned, and noisy, so we pivot (literally) to LSQR with regularization.

### 4.2 Regularization Tradeoff

Tikhonov regularization introduces a bias-variance tradeoff:
- **Too little damping**: noise is amplified (RMSE spikes)
- **Too much damping**: the solution is oversmoothed (clean RMSE increases from 0.022 to 0.083)

The optimal damp parameter (damp=2.0 for this system) balances these effects.

### 4.3 Limitations

- Small phantom size (32×32). Larger sizes require more memory and computation time
- Simple parallel-beam geometry (real CT uses cone-beam/fan-beam)
- No iterative reconstruction methods (e.g., ART, SIRT) for comparison
- DICOM JPEG Lossless decompression requires external libraries (`pylibjpeg`)

---

## 5. Conclusion

This project demonstrates end-to-end CT image reconstruction using:

1. A sparse forward model built from ray tracing
2. LSQR iterative least-squares solver for overdetermined systems
3. Tikhonov regularization for noise robustness
4. Comprehensive quality assessment with RMSE, PSNR, and SSIM

The system achieves high-quality reconstruction (RMSE 0.022, PSNR 33.4 dB) on synthetic phantoms and gracefully degrades under noise when regularization is applied. Real DICOM scans can be used as ground truth images for demonstration.

---

## 6. How to Run

```bash
pip install -r requirements.txt

# Basic reconstruction
python main.py reconstruct

# With iterative refinement
python main.py reconstruct --refine

# Using a real DICOM scan
python main.py reconstruct --input samples/CT-brain.dcm --compare

# Run all validations
python main.py validate --all

# Interactive menu
python main.py interactive
```

---

## 7. References

1. Shepp, L. A., & Logan, B. F. (1974). "The Fourier reconstruction of a head section." *IEEE Transactions on Nuclear Science*, 21(3), 21-43.
2. Paige, C. C., & Saunders, M. A. (1982). "LSQR: An algorithm for sparse linear equations and sparse least squares." *ACM Transactions on Mathematical Software*, 8(1), 43-71.
3. Hansen, P. C. (2010). *Discrete Inverse Problems: Insight and Algorithms*. SIAM.
4. Barre's DICOM Sample Collection — https://barre.dev/medical/samples/
5. Open Microscopy Environment — https://downloads.openmicroscopy.org/images/DICOM/samples/
