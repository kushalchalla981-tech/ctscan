"""Validation suite for all CT reconstruction components."""

import sys
import warnings
import numpy as np
from scipy.sparse import random as sparse_random, eye as sparse_eye

from src.phantom import shepp_logan
from src.projector import build_system, get_sparsity
from src.lud_solver import solve_lu, iterative_refinement
from src.metrics import compute_metrics, ssim
from src.noise import noise_robustness_test, add_gaussian_noise
from src.reconstructor import reconstruct


def utf8():
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')


# ── Phase 1: Forward Model ──────────────────────────────────────────────

def validate_forward_model(size: int = 32):
    results = []

    phantom = shepp_logan(size)
    ok = phantom.shape == (size, size) and 0 <= phantom.min() <= phantom.max() <= 1
    results.append(('Phantom shape and range', ok,
                    f'{phantom.shape}, range [{phantom.min():.3f}, {phantom.max():.3f}]'))

    A, b, x_true = build_system(size)
    results.append(('System matrix A shape', A.shape[0] > A.shape[1],
                    f'{A.shape} (overdetermined)'))
    results.append(('Sparsity > 80%', get_sparsity(A) > 0.80,
                    f'{get_sparsity(A):.2%}'))

    b_computed = A @ x_true
    residual = np.linalg.norm(b - b_computed) / np.linalg.norm(b)
    results.append(('Forward model residual < 1.1e-6', residual < 1.1e-6,
                    f'{residual:.2e}'))

    results.append(('Sinogram non-negative', np.all(b >= 0),
                    f'min={b.min():.4f}'))

    return results, {'A': A, 'b': b, 'x_true': x_true, 'phantom': phantom}


# ── Phase 2: LU Solver ──────────────────────────────────────────────────

def validate_lu_solver():
    results = []
    np.random.seed(42)

    # Dense system
    n = 10
    A = np.random.randn(n, n) + 5 * np.eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    x_sol, info = solve_lu(A, b, method='dense')
    err = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
    results.append(('Dense system (n=10) solution error < 1e-10', err < 1e-10,
                    f'{err:.2e}'))
    results.append(('Dense system residual < 1e-10', info['residual'] < 1e-10,
                    f'{info["residual"]:.2e}'))

    # Sparse system
    n = 100
    A_sp = sparse_random(n, n, density=0.05, format='csr')
    A_sp = A_sp + 2 * sparse_random(n, n, density=1.0, format='csr').multiply(np.eye(n))
    A_sp = A_sp.tocsr()
    x_true = np.random.randn(n)
    b = A_sp @ x_true
    x_sol, info = solve_lu(A_sp, b, method='sparse')
    err = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
    results.append(('Sparse system (n=100) error < 1e-8', err < 1e-8,
                    f'{err:.2e}'))
    results.append(('Sparse LU factorization', info['factorization'] == 'SuperLU',
                    info['factorization']))

    # Iterative refinement
    n = 50
    A = np.random.randn(n, n) + 4 * np.eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    x0, _ = solve_lu(A, b)
    x_ref, ref_info = iterative_refinement(A, b, x0, max_iter=3)
    err0 = np.linalg.norm(x0 - x_true) / np.linalg.norm(x_true)
    err_ref = np.linalg.norm(x_ref - x_true) / np.linalg.norm(x_true)
    results.append(('Refinement improves solution', err_ref <= err0 + 1e-15,
                    f'{err0:.2e} -> {err_ref:.2e}'))

    # Ill-conditioned warning
    n = 20
    A_ill = np.random.randn(n, n)
    A_ill[:, -1] = A_ill[:, 0] * 1.0001
    b = np.random.randn(n)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        solve_lu(A_ill, b, method='dense')
    results.append(('Ill-conditioned warning raised', len(w) > 0,
                    f'{len(w)} warning(s)'))

    return results


# ── Phase 3: Reconstruction ─────────────────────────────────────────────

def validate_reconstruction(size: int = 32):
    results = []

    res = reconstruct(size=size, use_refinement=False, method='auto')
    m = res['metrics']

    results.append(('RMSE < 0.05', m['rmse'] < 0.05, f'{m["rmse"]:.4f}'))
    results.append(('PSNR > 25 dB', m['psnr'] > 25, f'{m["psnr"]:.2f} dB'))
    results.append(('Residual < 1e-3', m['residual'] < 1e-3,
                    f'{m["residual"]:.2e}'))
    results.append(('SSIM > 0.95', m['ssim'] > 0.95, f'{m["ssim"]:.4f}'))
    results.append(('Image shapes match', res['phantom'].shape == res['reconstruction'].shape,
                    f'{res["phantom"].shape}'))

    res_ref = reconstruct(size=size, use_refinement=True, method='auto')
    m_ref = res_ref['metrics']
    results.append(('Refinement residual <= baseline', m_ref['residual'] <= m['residual'] * 1.1,
                    f'{m["residual"]:.2e} -> {m_ref["residual"]:.2e}'))

    return results


# ── Phase 4: Noise Robustness ───────────────────────────────────────────

def validate_noise_robustness(size: int = 32):
    results = []

    unreg = noise_robustness_test(size=size, noise_levels=[0.0, 0.01],
                                  use_regularization=False, seed=42)
    f1 = unreg[0.01]['rmse'] > 10 * unreg[0.0]['rmse']
    results.append(('Unregularized LSQR fails at 1% noise', f1,
                    f'RMSE {unreg[0.0]["rmse"]:.4f} -> {unreg[0.01]["rmse"]:.1f}'))

    reg = noise_robustness_test(size=size, noise_levels=[0.0, 0.01, 0.05],
                                use_regularization=True, seed=42)
    f2 = reg[0.05]['ssim'] > 0.3
    results.append(('Regularized: 5% noise SSIM > 0.3', f2,
                    f'SSIM={reg[0.05]["ssim"]:.4f}'))

    rmses = [reg[nl]['rmse'] for nl in [0.0, 0.01, 0.05]]
    f3 = all(rmses[i] <= rmses[i + 1] + 1e-6 for i in range(len(rmses) - 1))
    results.append(('Regularized: monotonic RMSE degradation', f3,
                    f'{" -> ".join(f"{v:.4f}" for v in rmses)}'))

    for nl in [0.01, 0.05]:
        u = noise_robustness_test(size=16, noise_levels=[nl],
                                  use_regularization=False, seed=42)
        r = noise_robustness_test(size=16, noise_levels=[nl],
                                  use_regularization=True, seed=42)
        f4 = r[nl]['rmse'] < u[nl]['rmse']
        results.append((f'Regularization helps at {nl*100:.0f}% noise', f4,
                        f'{u[nl]["rmse"]:.4f} -> {r[nl]["rmse"]:.4f}'))

    return results


# ── Run All ─────────────────────────────────────────────────────────────

def run_all(size: int = 32):
    results = {}

    fwd, data = validate_forward_model(size)
    results['Forward Model'] = fwd

    lu = validate_lu_solver()
    results['LU Solver'] = lu

    rec = validate_reconstruction(size)
    results['Reconstruction'] = rec

    noise = validate_noise_robustness(size)
    results['Noise Robustness'] = noise

    return results
