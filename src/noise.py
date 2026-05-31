"""Noise addition for CT sinogram robustness testing."""

import numpy as np
from typing import Optional


def add_gaussian_noise(
    b: np.ndarray,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to a sinogram.

    Noise is proportional to the signal magnitude:
        b_noisy = b + noise_level * std(b) * N(0,1)

    Args:
        b: Clean sinogram vector
        noise_level: Relative noise level (e.g., 0.05 = 5%)
        seed: Random seed for reproducibility

    Returns:
        Noisy sinogram vector
    """
    if seed is not None:
        np.random.seed(seed)

    sigma = noise_level * np.std(b)
    noise = np.random.randn(*b.shape).astype(b.dtype) * sigma

    return b + noise


def add_poisson_noise(
    b: np.ndarray,
    photon_count: float = 1e5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Poisson (photon) noise to a sinogram.

    Models quantum noise in X-ray detection:
        b_noisy ~ Poisson(photon_count * exp(-b)) / photon_count

    Args:
        b: Clean sinogram vector
        photon_count: Incident photon count (higher = less noise)
        seed: Random seed for reproducibility

    Returns:
        Noisy sinogram vector
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize b to attenuation-like values
    b_norm = b - b.min()
    b_norm = b_norm / (b_norm.max() + 1e-10)

    # Simulate photon counting
    detected = np.random.poisson(photon_count * np.exp(-b_norm))
    noisy = -np.log(np.maximum(detected, 1) / photon_count)

    # Rescale back to original range
    noisy = noisy * (b.max() - b.min()) + b.min()

    return noisy.astype(b.dtype)


def noise_robustness_test(
    size: int = 32,
    noise_levels: list = None,
    use_refinement: bool = False,
    use_regularization: bool = False,
    regularization_strength: float = 4.0,
    seed: int = 42
) -> dict:
    """
    Run reconstruction at multiple noise levels and collect metrics.

    Args:
        size: Phantom dimension
        noise_levels: List of relative noise levels to test
        use_refinement: Apply iterative refinement (harmful with noise, for demo only)
        use_regularization: Apply Tikhonov regularization
        regularization_strength: Regularization parameter (lambda in damp=sqrt(lambda)).
                                 Use 4.0 for damp=2.0 (recommended for CT noise).
        seed: Random seed

    Returns:
        Dictionary mapping noise level -> metrics dict
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]

    from src.projector import build_system
    from src.lud_solver import solve_lu, iterative_refinement
    from src.metrics import compute_metrics, ssim

    A, b_clean, x_true = build_system(size)
    results = {}

    for nl in noise_levels:
        np.random.seed(seed)
        b_noisy = add_gaussian_noise(b_clean, nl)

        regularization = regularization_strength if use_regularization else None
        x_rec, info = solve_lu(A, b_noisy, regularization=regularization)

        if use_refinement:
            x_rec, ref_info = iterative_refinement(A, b_noisy, x_rec, max_iter=3)

        metrics = compute_metrics(x_true, x_rec, A, b_noisy)
        metrics['ssim'] = ssim(x_true, x_rec, size)
        metrics['noise_level'] = nl
        metrics['solver_info'] = info

        results[nl] = metrics

    return results
