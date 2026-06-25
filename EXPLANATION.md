# CT Reconstruction using LU Decomposition — Student Explanation

## To My Teacher: What I Built and How It Works

---

## The Big Idea in One Paragraph

I built a program that **simulates what happens inside a CT (Computed Tomography) scanner** — but only the mathematical part. A real CT scanner sends X-rays through a patient's body from many different angles, measures how much radiation gets through, and then uses math to figure out what the inside of the body looks like. My program does the same thing, but instead of a real patient, I start with a known image (called a **phantom**), simulate the X-ray measurements, and then try to reconstruct the original image from those measurements. The whole point is to **study the reconstruction algorithm** — how well can we recover the original image, and what happens when things go wrong (noise, bad data, etc.).

---

## Why This Matters

CT scans are one of the most important medical imaging technologies. Every time someone gets a CT scan, a computer is solving a problem that looks like this:

```
Given the measurements (sinogram), find the original image
```

This is called an **inverse problem**. The direct problem is easy: if I know what's inside, I can simulate the X-ray measurements. The inverse problem (reconstructing the image from measurements) is much harder. My project explores exactly this challenge using **linear algebra** — specifically LU decomposition and least-squares solvers.

---

## The Four Phases of the Project

### Phase 1: The Forward Model (Simulating X-rays)

**What I did:**
I created a mathematical model of how X-rays pass through an object. I used the **Shepp-Logan phantom** — a standard test image in medical imaging that looks like a cross-section of a human head with different tissues (skull, brain, ventricles).

**The math:**
I set up a system of equations:

```
A · x = b
```

Where:
- `x` is the image (unknown, what we want to find) — a vector of pixel values
- `A` is the **system matrix** — each row represents one X-ray beam going through the image at a specific angle and position
- `b` is the **sinogram** — the measurements from the detectors

Each entry in matrix `A` represents how long a particular X-ray beam travels through a particular pixel. I built this using **parallel-beam geometry**: I simulate X-ray beams at 44 angles (0° to 180°) and 32 detector positions per angle, giving me `44 × 32 = 1408` measurements for a `32 × 32 = 1024` pixel image.

**Key result:** The matrix `A` is about **91% sparse** — most entries are zero because each X-ray beam only passes through a small fraction of the total pixels. This is critical because it means we can use **sparse linear algebra** techniques that are much faster than treating it as a dense matrix. The matrix has about 127,000 non-zero entries out of 1.4 million total.

```
A shape: (1408, 1024)    ← overdetermined system (more equations than unknowns)
Sparsity: 91.16%          ← mostly zeros
Non-zeros: ~127,000
```

---

### Phase 2: The LU Solver (Solving the System)

**What I did:**
I implemented algorithms to solve `A · x = b` — that is, find the image `x` that best explains the measurements `b`.

**The math:**
Since `A` is not square (1408 equations, 1024 unknowns), there is no exact solution. Instead, we find the **least-squares solution** — the `x` that minimizes `‖A·x — b‖²` (the sum of squared differences between predicted and actual measurements).

I implemented two approaches:

1. **Dense LU decomposition** — for small systems, I factor `A` into `P·L·U` where `L` is lower triangular, `U` is upper triangular, and `P` is a permutation matrix. Then solving `A·x = b` becomes two easy steps: forward-substitution then back-substitution. But for our 32×32 phantom, the normal equations `Aᵀ·A·x = Aᵀ·b` produce a matrix that is dense and 1024×1024 — which is workable but not ideal.

2. **Sparse LSQR** — this is an **iterative method** (like Conjugate Gradient) that works directly with the sparse matrix `A` without ever forming `Aᵀ·A`. It is vastly more efficient for large sparse systems. In practice, my program automatically detects which method to use based on the problem size.

**Iterative refinement:** I also implemented **iterative refinement** — after getting an initial solution, I compute the residual `r = b — A·x`, solve `A·Δx = r`, and add the correction `x = x + Δx`. This dramatically improves accuracy:

| Metric | Basic LSQR | With Refinement |
|--------|-----------|-----------------|
| RMSE | 0.0223 | 0.0012 |
| PSNR | 33.38 dB | 58.35 dB |
| SSIM | 0.995 | 1.000 |
| Residual | 2.6×10⁻⁵ | 2.3×10⁻⁷ |

The key insight: iterative refinement works because the residual is smaller than the original right-hand side, so the correction step is more accurate.

---

### Phase 3: Reconstruction and Quality Metrics

**What I did:**
I put it all together — take a phantom, simulate X-ray projections, reconstruct the image, and measure how good the reconstruction is.

**The metrics I use to measure quality:**

1. **RMSE (Root Mean Square Error):** How different is each pixel on average? Lower is better.
   — `RMSE = sqrt(mean((x_true — x_rec)²))`

2. **PSNR (Peak Signal-to-Noise Ratio):** The ratio between the maximum possible signal and the error. Higher is better, measured in decibels (dB).
   — `PSNR = 20 · log₁₀(max / RMSE)`

3. **SSIM (Structural Similarity Index):** Compares patterns of pixel intensity — it looks at **structure**, not just per-pixel differences. This is the most perceptually relevant metric. A value of 1.0 means identical.
   — SSIM considers luminance, contrast, and structure separately

4. **Relative error** and **residual:** Additional checks on how well we solved the system

**Results with the Shepp-Logan phantom (32×32, clean data):**
```
RMSE:   0.0223    (2% average pixel error)
PSNR:  33.38 dB   (good quality, >25 dB is the target)
SSIM:  0.995      (nearly identical structure)
```

With refinement, these improve to RMSE 0.0012, PSNR 58.35 dB, SSIM 1.000 — essentially a perfect reconstruction when there is no noise.

---

### Phase 4: Noise Robustness (Real-World Reality)

**What I did:**
In the real world, CT measurements are **noisy** — the X-ray detector has electronic noise, the patient moves, and there are random quantum fluctuations in the X-ray beam itself. I added Gaussian noise to the sinogram and measured how the reconstruction degrades.

**What happens without regularization:**
At just 1% noise, the reconstruction **catastrophically fails**:
```
Clean:    RMSE 0.0223    (excellent)
1% noise: RMSE 2.5600    (garbage — worse than random)
```

Why? Because the system `A` is **ill-conditioned** — small changes in `b` cause huge changes in `x`. This is a fundamental property of inverse problems: they amplify noise.

**The fix: Tikhonov Regularization**
Instead of solving `min ‖A·x — b‖²`, I solve:
```
min ‖A·x — b‖² + λ² · ‖x‖²
```

The second term penalizes large solution values, which prevents the noise from blowing up. The parameter `λ` (called `damp` in LSQR) controls the trade-off: too little and noise still causes problems, too much and the solution becomes overly smooth (bias).

**Results with regularization (damp = 2.0):**
| Noise | RMSE | PSNR | SSIM |
|-------|------|------|------|
| 0% | 0.083 | 21.6 dB | 0.913 |
| 1% | 0.095 | 20.4 dB | 0.897 |
| 5% | 0.245 | 14.7 dB | 0.551 |
| 10% | 0.467 | 12.1 dB | 0.252 |
| 20% | 0.922 | 10.6 dB | 0.080 |

At 0% noise, regularization adds a small bias (RMSE 0.083 vs 0.022 without it). But at 1% noise, regularization keeps RMSE at 0.095 instead of the catastrophic 2.56. **This trade-off — bias vs. variance — is the central tension in all inverse problems.**

---

## What I Learned

### 1. Inverse problems are fundamentally hard
The forward problem (simulate X-rays through known image) is trivial. The inverse problem (find image from X-ray measurements) is **ill-posed** — tiny measurement errors become huge reconstruction errors. This is why medical CT scanners use sophisticated algorithms, specialized hardware, and radiation dose optimization.

### 2. Sparsity is your friend
Matrix `A` is 91% sparse. Exploiting this sparsity with iterative solvers (LSQR) instead of dense factorization is what makes the problem computationally feasible at larger sizes. At 32×32 it doesn't matter much, but at clinical resolutions (512×512) it's the difference between seconds and hours.

### 3. Regularization is essential
Without regularization, even 1% noise destroys the reconstruction. With Tikhonov regularization, the system becomes robust up to about 5-10% noise. The optimal trade-off depends on the noise level — and in a real scanner, you don't know the noise level in advance.

### 4. LSQR vs. LU: different tools for different jobs
- **LU decomposition** is exact (to machine precision) but requires forming `Aᵀ·A`, which is dense and expensive
- **LSQR** is iterative and approximate, but can handle sparse systems directly and can incorporate regularization naturally
- My program auto-selects between them, but LSQR with regularization is the practical choice for real problems

---

## The CLI: How to Use What I Built

The program has six main commands:

```
python main.py reconstruct        # Reconstruct the Shepp-Logan phantom
python main.py upload              # Pick any image → simulate CT reconstruction
python main.py validate --all     # Run all validation checks
python main.py noise --regularize # Test noise robustness
python main.py interactive        # Menu-driven mode (no flags needed)
python main.py info               # Project information
```

The `upload` command is especially useful for demonstrations — you pick any image file (DICOM, PNG, JPG), and the program:
1. Resizes it to use as the ground truth phantom
2. Simulates X-ray projections through it
3. Reconstructs it from those projections
4. Shows you how accurate the reconstruction was

This makes the abstract math concrete — you can see with your own eyes how the algorithm performs on different types of images.

---

## The Math Behind It All (Formal Summary)

For a complete formal treatment:

1. **Forward model:** `A · x = b` where `A ∈ ℝ^{m×n}`, `m > n`
2. **Least-squares solution:** `x̂ = argmin ‖A·x — b‖²`
3. **Normal equations:** `(Aᵀ·A)·x = Aᵀ·b`
4. **LU decomposition:** `Aᵀ·A = P·L·U`, then forward/back-substitution
5. **LSQR (iterative):** Bidiagonalization + Golub-Kahan process, equivalent to CG on the normal equations
6. **Iterative refinement:** `x_{k+1} = x_k + A†·(b — A·x_k)`
7. **Tikhonov regularization:** `x̂ = argmin ‖A·x — b‖² + λ²‖x‖²`
8. **Quality metrics:** `RMSE`, `PSNR`, `SSIM` for comparing `x̂` to true `x`

---

## References

- Shepp, L. A., & Logan, B. F. (1974). "The Fourier reconstruction of a head section." *IEEE Transactions on Nuclear Science*, 21(3), 21-43.
- Paige, C. C., & Saunders, M. A. (1982). "LSQR: An algorithm for sparse linear equations and sparse least squares." *ACM Transactions on Mathematical Software*, 8(1), 43-71.
- Tikhonov, A. N., & Arsenin, V. Y. (1977). *Solutions of Ill-Posed Problems*. Winston & Sons.
- Wang, Z., et al. (2004). "Image quality assessment: from error visibility to structural similarity." *IEEE Transactions on Image Processing*, 13(4), 600-612.
