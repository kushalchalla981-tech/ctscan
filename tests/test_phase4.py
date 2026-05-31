"""Unit tests for Phase 4 noise robustness."""

import numpy as np
import pytest
from src.noise import add_gaussian_noise, add_poisson_noise, noise_robustness_test
from src.projector import build_system


def test_gaussian_noise_shape():
    """Test noise preserves sinogram shape."""
    _, b, _ = build_system(16)
    b_noisy = add_gaussian_noise(b, noise_level=0.05, seed=42)
    assert b_noisy.shape == b.shape
    assert b_noisy.dtype == b.dtype


def test_gaussian_noise_zero_level():
    """Test zero noise returns identical sinogram."""
    _, b, _ = build_system(16)
    b_noisy = add_gaussian_noise(b, noise_level=0.0, seed=42)
    np.testing.assert_array_equal(b_noisy, b)


def test_gaussian_noise_increases_error():
    """Test that noise increases forward model error."""
    A, b, x_true = build_system(16)
    clean_res = np.linalg.norm(A @ x_true - b)
    b_noisy = add_gaussian_noise(b, noise_level=0.10, seed=42)
    noisy_res = np.linalg.norm(A @ x_true - b_noisy)
    assert noisy_res > clean_res, "Noise did not increase measurement error"


def test_gaussian_noise_reproducibility():
    """Test that seed gives reproducible noise."""
    _, b, _ = build_system(16)
    b1 = add_gaussian_noise(b, noise_level=0.05, seed=123)
    b2 = add_gaussian_noise(b, noise_level=0.05, seed=123)
    np.testing.assert_array_equal(b1, b2)


def test_gaussian_noise_different_seeds():
    """Test different seeds give different noise."""
    _, b, _ = build_system(16)
    b1 = add_gaussian_noise(b, noise_level=0.05, seed=123)
    b2 = add_gaussian_noise(b, noise_level=0.05, seed=456)
    assert not np.allclose(b1, b2)


def test_poisson_noise_shape():
    """Test Poisson noise preserves shape."""
    _, b, _ = build_system(16)
    b_noisy = add_poisson_noise(b, photon_count=1e5, seed=42)
    assert b_noisy.shape == b.shape


def test_poisson_noise_photon_count():
    """Test lower photon count increases noise."""
    _, b, _ = build_system(16)
    b_high = add_poisson_noise(b, photon_count=1e6, seed=42)
    b_low = add_poisson_noise(b, photon_count=1e3, seed=42)
    err_high = np.linalg.norm(b_high - b)
    err_low = np.linalg.norm(b_low - b)
    assert err_low > err_high, "Lower photon count should increase noise"


def test_robustness_test_structure():
    """Test that robustness test returns expected structure."""
    results = noise_robustness_test(size=16, noise_levels=[0.0, 0.05], seed=42)
    assert 0.0 in results
    assert 0.05 in results
    for nl, metrics in results.items():
        assert 'rmse' in metrics
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'residual' in metrics
        assert 'noise_level' in metrics


def test_unregularized_catastrophic_at_1pct():
    """Test that unregularized LSQR fails catastrophically at 1% noise."""
    results = noise_robustness_test(
        size=32, noise_levels=[0.0, 0.01],
        use_regularization=False, seed=42
    )
    # 1% noise should cause RMSE >> clean RMSE for unregularized solver
    rmse_0 = results[0.0]['rmse']
    rmse_1 = results[0.01]['rmse']
    assert rmse_1 > 10 * rmse_0, \
        f"Unregularized LSQR should degrade sharply: {rmse_0:.4f} -> {rmse_1:.4f}"


def test_regularized_monotonic_degradation():
    """Test that regularized RMSE increases monotonically with noise."""
    results = noise_robustness_test(
        size=16, noise_levels=[0.0, 0.01, 0.05, 0.10],
        use_regularization=True, seed=42
    )
    rmses = [results[nl]['rmse'] for nl in [0.0, 0.01, 0.05, 0.10]]
    for i in range(len(rmses) - 1):
        assert rmses[i] <= rmses[i + 1] + 1e-6, \
            f"RMSE not monotonic at step {i}: {rmses[i]} > {rmses[i+1]}"


def test_regularized_quality_at_5pct():
    """Test regularized reconstruction is recognizable at 5% noise."""
    results = noise_robustness_test(
        size=32, noise_levels=[0.05],
        use_regularization=True, seed=42
    )
    # At 5% noise, structure should still be recognizable
    assert results[0.05]['ssim'] > 0.3, \
        f"5% noise SSIM too low: {results[0.05]['ssim']}"


def test_regularized_better_than_unregularized():
    """Test that regularization always helps at non-zero noise."""
    for nl in [0.01, 0.05, 0.10]:
        unreg = noise_robustness_test(
            size=16, noise_levels=[nl], use_regularization=False, seed=42
        )
        reg = noise_robustness_test(
            size=16, noise_levels=[nl], use_regularization=True, seed=42
        )
        assert reg[nl]['rmse'] < unreg[nl]['rmse'], \
            f"Regularization should improve RMSE at {nl*100}% noise"
