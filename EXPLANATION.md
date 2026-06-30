# LU Decomposition in CT Reconstruction — A Complete Walkthrough

## 1. The Big Picture: What Are We Actually Solving?

A CT scanner measures how much X-ray intensity is lost as beams pass through an object from many angles. Each measurement gives us one linear equation:

```
(ray contribution from pixel 1) + (pixel 2) + ... + (pixel N) = measurement
```

Stacking all measurements gives us the **linear system**:

```
A × x = b
```

| Symbol | Size | Meaning |
|--------|------|---------|
| **A** | `m × n` | System matrix — each row is one X-ray beam, each column is one pixel |
| **x** | `n × 1` | Unknown image (what we want to recover) |
| **b** | `m × 1` | Measured sinogram data (the CT scanner output) |

For a `32 × 32` image:  
- **n** = 1024 pixels  
- **m** = 32 angles × 45 detectors = 1440 measurements  

Since **m > n**, the system is **overdetermined** (more equations than unknowns). There is no exact solution — we must find the **best approximation** in the least-squares sense.

---

## 2. How A Is Built — The Forward Model (`src/projector.py`)

The matrix **A** is constructed by simulating X-ray beams through a pixel grid:

```
For each angle θ (0° to 180°):
    For each detector position d:
        Fire a ray through the grid
        For each pixel (i, j) the ray intersects:
            A[row, pixel_idx] = intersection length of the ray through that pixel
            b[row] += (intersection length) × (pixel value)
```

Each entry `A[row, col]` is the **length of the ray's path through pixel `col`**.  
The result is a **sparse matrix** (~96% zeros for 32×32) because each ray hits only a few pixels out of the entire grid.

---

## 3. LU Decomposition — The Core Algorithm (`src/lud_solver.py`)

### 3.1 What is LU Decomposition?

LU decomposition factors a square matrix **A** into:

```
A = L × U
```

Where:  
- **L** = lower triangular matrix (ones on diagonal)  
- **U** = upper triangular matrix  

Once factored, solving `Ax = b` becomes two easy steps:

```
1. Solve L y = b  (forward substitution)
2. Solve U x = y  (backward substitution)
```

### 3.2 Why Partial Pivoting? (PA = LU)

Naive LU fails when a pivot element is zero or very small. **Partial pivoting** fixes this by swapping rows at each step:

```
At step k, find the row p (≥ k) with the largest |A[p, k]|
Swap row k and row p
Proceed with elimination
```

This gives us the factorization:

```
P × A = L × U
```

Where **P** is a permutation matrix encoding the row swaps.

### 3.3 The Algorithm Step-by-Step (from `_dense_lu_solve`)

For each column `k` from 0 to `n-2`:

```
Step 1 — Find pivot:
    pivot_row = k + argmax(|A[k:n, k]|)

Step 2 — Check singularity:
    if |A[pivot_row, k]| < 1e-14 → matrix is singular, abort

Step 3 — Swap rows (if needed):
    swap A[k] ↔ A[pivot_row]
    swap b[k] ↔ b[pivot_row]
    swap P[k] ↔ P[pivot_row]

Step 4 — Eliminate below:
    scale factor:  A[k+1:n, k] /= A[k, k]
    rank-1 update: A[k+1:n, k+1:n] -= outer(A[k+1:n, k], A[k, k+1:n])
```

After the loop, **A** is overwritten with **L** (lower part) and **U** (upper part, including diagonal).

### 3.4 Forward Substitution (Ly = Pb)

Solve for y from top to bottom:

```
for i = 0 to n-1:
    y[i] = b[P[i]] - Σ(A[i, j] × y[j] for j < i)
```

### 3.5 Backward Substitution (Ux = y)

Solve for x from bottom to top:

```
for i = n-1 down to 0:
    x[i] = (y[i] - Σ(A[i, j] × x[j] for j > i)) / A[i, i]
```

---

## 4. Handling Overdetermined Systems — LSQR

Since the CT system **A** is rectangular (m > n), we cannot apply LU directly. Instead, the project solves the **least-squares problem**:

```
minimize  ‖Ax − b‖²
```

This is equivalent to solving the **normal equations**:

```
(A^T A) x = A^T b
```

The project uses **LSQR** (an iterative Krylov-subspace method) for sparse matrices via `scipy.sparse.linalg.lsqr`, and `numpy.linalg.lstsq` for dense matrices.

### Why not form A^T A explicitly?

For a 32×32 image:  
- **A** is 1440 × 1024 → ~1.5M entries (sparse)  
- **A^T A** is 1024 × 1024 → ~1M entries (dense)  

Forming A^T A explicitly is expensive and numerically less stable. LSQR works directly with **A**, avoiding this.

---

## 5. Tikhonov Regularization — Handling Noise

Real CT data has measurement noise. The unregularized least-squares solution can be wildly inaccurate. **Tikhonov regularization** adds a penalty term:

```
minimize  ‖Ax − b‖² + λ ‖x‖²
```

This is equivalent to solving:

```
(A^T A + λ I) x = A^T b
```

In LSQR, this is done via the `damp` parameter: `damp = √λ`.  
The effect: **small singular values are damped**, preventing noise from blowing up the solution.

From the validation results:
| Noise | Unregularized RMSE | Regularized RMSE |
|-------|-------------------|------------------|
| 0%    | ~0.003            | ~0.003           |
| 1%    | >0.1 (fails)      | ~0.02            |
| 5%    | N/A               | ~0.05            |

Without regularization, even 1% noise destroys the reconstruction.

---

## 6. Iterative Refinement (`src/lud_solver.py:156`)

For extra accuracy on square systems, **iterative refinement** improves the solution:

```
for i = 0 to max_iter:
    r = b − A x                    # compute residual
    if ‖r‖ / ‖b‖ < 1e-10: break   # stop if accurate enough
    solve A Δx = r                 # solve for correction (reuses LU)
    x += Δx                        # apply correction
```

This is useful when the matrix is ill-conditioned and the initial LU solution has significant round-off error. The same LU factors can be reused for each correction step.

---

## 7. Complete CT Pipeline (`src/reconstructor.py`)

```
Input: phantom size, noise level, method choice

1. Build A and b via ray-tracing  (projector.py)
   → Sparse matrix A (1440 × 1024)
   → Sinogram vector b

2. (Optional) Add Gaussian noise  (noise.py)
   → b_noisy = b + noise_level × σ(b) × N(0,1)

3. Solve Ax ≈ b  (lud_solver.py)
   → Overdetermined → LSQR (with Tikhonov if requested)
   → Square → PA=LU → forward/backward substitution

4. (Optional) Iterative refinement
   → Solve A Δx = b − Ax, update x

5. Compute quality metrics  (metrics.py)
   → RMSE, PSNR, SSIM, relative error, residual
```

---

## 8. How It All Fits Together — A Concrete Example

For `size=32` with default parameters:

```
1. phantom = shepp_logan(32)           → 32×32 image
2. A, b, x_true = build_system(32)     → A: 1440×1024 (sparse, 96% zeros)
                                           b: 1440×1
3. x_rec, info = solve_lu(A, b)        → LSQR solver (auto-detected)
4. reshape x_rec → 32×32 image
5. Compare with phantom:
   RMSE  ≈ 0.003    (target: < 0.05)
   PSNR  ≈ 45 dB    (target: > 25 dB)
   SSIM  ≈ 0.998    (target: > 0.95)
   Residual ≈ 1e-15 (target: < 1e-3)
```

---

## 9. Numerical Considerations and Pitfalls

| Issue | Cause | Mitigation |
|-------|-------|------------|
| **Singular matrix** | Pivot too small during LU | Check `|A[k,k]| > 1e-14`, abort if singular |
| **Ill-conditioned** | Large condition number `κ(A) > 1e10` | Warn user; use regularization |
| **Noise amplification** | Small singular values blow up | Tikhonov damping `damp = √λ` |
| **Sparsity** | ~96% zeros in A | Use sparse matrix formats (LIL → CSR/CSC) |
| **Memory** | 1440×1024 dense → 11.5 MB; sparse → ~0.5 MB | Keep A sparse throughout |

---

## 10. Key Source Files Summary

| File | Purpose |
|------|---------|
| `src/projector.py` | Builds sparse system matrix **A** via ray-tracing |
| `src/lud_solver.py` | Implements PA=LU, LSQR, iterative refinement |
| `src/reconstructor.py` | Orchestrates the full pipeline |
| `src/phantom.py` | Generates Shepp-Logan test phantom |
| `src/metrics.py` | RMSE, PSNR, SSIM computation |
| `src/noise.py` | Gaussian/Poisson noise + robustness sweeps |
| `src/fbp_solver.py` | Filtered Back Projection (alternative method) |
| `src/validate.py` | 4-phase validation suite |
| `main.py` | CLI entry point with 8 subcommands |

---

## 11. Quick Reference — Running the Code

```bash
# Basic reconstruction (LSQR)
python main.py reconstruct --size 32

# With Tikhonov regularization + 5% noise
python main.py reconstruct --size 32 --noise 5 --regularize

# Filtered Back Projection
python main.py reconstruct --size 32 --method fbp --filter shepp-logan

# 3D volume (stack of slices)
python main.py reconstruct-3d --depth 16 --size 32 --method fbp

# Upload your own image
python main.py upload

# Run all validations
python main.py validate --all

# Noise robustness test
python main.py noise --levels 0 1 5 10 --regularize --plot noise.png

# Interactive menu
python main.py interactive
```

---

*This project was built as an educational prototype for CT reconstruction using linear algebra. The core numerical work is solving a large, sparse, overdetermined linear system — first by recognizing it as a least-squares problem, then applying iterative solvers (LSQR) with Tikhonov regularization to recover a stable, accurate image from noisy projection measurements.*
