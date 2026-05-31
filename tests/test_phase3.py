"""Unit tests for Phase 3 reconstruction."""

import numpy as np
import pytest
from src.reconstructor import reconstruct
from src.metrics import compute_metrics, ssim


def test_reconstruction_basic():
    """Test basic reconstruction pipeline."""
    results = reconstruct(size=16, use_refinement=False, method='auto')
    
    assert 'phantom' in results
    assert 'reconstruction' in results
    assert 'metrics' in results
    assert results['phantom'].shape == (16, 16)
    assert results['reconstruction'].shape == (16, 16)


def test_reconstruction_quality():
    """Test reconstruction meets quality criteria."""
    results = reconstruct(size=32, use_refinement=False, method='auto')
    metrics = results['metrics']
    
    assert metrics['rmse'] < 0.05, f"RMSE too high: {metrics['rmse']}"
    assert metrics['psnr'] > 25, f"PSNR too low: {metrics['psnr']}"
    assert metrics['residual'] < 1e-3, f"Residual too high: {metrics['residual']}"


def test_metrics_computation():
    """Test metrics computation."""
    x_true = np.random.rand(100)
    x_rec = x_true + 0.01 * np.random.randn(100)
    
    metrics = compute_metrics(x_true, x_rec)
    
    assert 'rmse' in metrics
    assert 'psnr' in metrics
    assert 'relative_error' in metrics
    assert metrics['rmse'] > 0
    assert metrics['psnr'] > 0


def test_ssim_perfect():
    """Test SSIM on identical images."""
    x = np.random.rand(256)
    ssim_val = ssim(x, x, 16)
    assert abs(ssim_val - 1.0) < 0.01, "SSIM should be ~1 for identical images"


def test_ssim_different():
    """Test SSIM on different images."""
    x1 = np.random.rand(256)
    x2 = np.random.rand(256)
    ssim_val = ssim(x1, x2, 16)
    assert ssim_val < 1.0, "SSIM should be <1 for different images"


def test_reconstruction_with_refinement():
    """Test reconstruction with iterative refinement."""
    results = reconstruct(size=16, use_refinement=True, method='auto')
    
    assert results['metrics']['residual'] < 1e-3
