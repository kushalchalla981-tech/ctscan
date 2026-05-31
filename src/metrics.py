"""Metrics for reconstruction quality assessment."""

import numpy as np
from typing import Dict


def compute_metrics(x_true: np.ndarray, x_rec: np.ndarray, A=None, b=None) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        x_true: Ground truth image (flattened)
        x_rec: Reconstructed image (flattened)
        A: System matrix (optional, for residual)
        b: Sinogram (optional, for residual)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # RMSE (normalized)
    mse = np.mean((x_true - x_rec) ** 2)
    rmse = np.sqrt(mse)
    metrics['rmse'] = rmse
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if rmse > 0:
        max_val = max(x_true.max(), x_rec.max())
        psnr = 20 * np.log10(max_val / rmse)
        metrics['psnr'] = psnr
    else:
        metrics['psnr'] = np.inf
    
    # Relative error
    rel_error = np.linalg.norm(x_true - x_rec) / np.linalg.norm(x_true)
    metrics['relative_error'] = rel_error
    
    # Forward model residual
    if A is not None and b is not None:
        residual = np.linalg.norm(A @ x_rec - b) / np.linalg.norm(b)
        metrics['residual'] = residual
    
    return metrics


def ssim(x_true: np.ndarray, x_rec: np.ndarray, size: int) -> float:
    """
    Compute Structural Similarity Index (simplified).
    
    Args:
        x_true: Ground truth (flattened)
        x_rec: Reconstruction (flattened)
        size: Image dimension
        
    Returns:
        SSIM value [0, 1]
    """
    img1 = x_true.reshape(size, size)
    img2 = x_rec.reshape(size, size)
    
    # Constants for stability
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_val
