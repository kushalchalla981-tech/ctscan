"""Unit tests for Phase 1 components."""

import numpy as np
import pytest
from src.phantom import shepp_logan
from src.projector import build_system, get_sparsity


def test_phantom_generation():
    """Test Shepp-Logan phantom generation."""
    phantom = shepp_logan(32)
    assert phantom.shape == (32, 32)
    assert 0 <= phantom.min() <= phantom.max() <= 1
    assert phantom.dtype == np.float64


def test_phantom_sizes():
    """Test phantom generation at different sizes."""
    for size in [16, 32, 64]:
        phantom = shepp_logan(size)
        assert phantom.shape == (size, size)


def test_system_matrix_shape():
    """Test system matrix dimensions."""
    size = 16
    A, b, x_true = build_system(size)
    
    n_pixels = size * size
    n_detectors = int(size * 1.4)
    n_angles = size
    n_measurements = n_angles * n_detectors
    
    assert A.shape == (n_measurements, n_pixels)
    assert b.shape == (n_measurements,)
    assert x_true.shape == (n_pixels,)


def test_system_sparsity():
    """Test that system matrix is highly sparse."""
    A, _, _ = build_system(16)
    sparsity = get_sparsity(A)
    assert sparsity > 0.80, f"Expected >80% sparsity, got {sparsity:.2%}"


def test_forward_model_consistency():
    """Test that Ax = b holds for ground truth."""
    A, b, x_true = build_system(16)
    b_computed = A @ x_true
    residual = np.linalg.norm(b - b_computed) / np.linalg.norm(b)
    assert residual < 1e-5, f"Forward model inconsistent: residual={residual:.2e}"


def test_non_negative_sinogram():
    """Test that sinogram values are non-negative."""
    _, b, _ = build_system(16)
    assert np.all(b >= 0), "Sinogram contains negative values"
