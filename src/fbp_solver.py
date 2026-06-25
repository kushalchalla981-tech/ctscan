"""Filtered Back Projection (FBP) reconstruction using the inverse Radon transform."""

import numpy as np
from skimage.transform import radon, iradon
from typing import Tuple, Optional


def fbp_forward(
    image: np.ndarray,
    n_angles: int = None,
    n_detectors: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward projection using the Radon transform.

    Args:
        image: 2D ground truth array (size x size)
        n_angles: Number of projection angles (default: image width)
        n_detectors: Number of detector bins (default: int(size * 1.4))

    Returns:
        sinogram: Flattened sinogram vector
        theta: Angle array in degrees
    """
    size = image.shape[0]
    if n_angles is None:
        n_angles = size
    if n_detectors is None:
        n_detectors = int(size * 1.4)

    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(image, theta=theta, circle=False)
    return sino.flatten(), theta


def fbp_reconstruct(
    image: np.ndarray,
    n_angles: int = None,
    filter: str = 'ramp',
    noise_level: float = 0.0,
    seed: Optional[int] = None
) -> dict:
    """
    Full FBP reconstruction pipeline: forward project, optionally add noise,
    then reconstruct using filtered back projection.

    Args:
        image: 2D ground truth array (size x size)
        n_angles: Number of projection angles (default: image width)
        filter: FBP filter type ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann')
        noise_level: Relative Gaussian noise added to sinogram (0 = no noise)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
            'phantom': ground truth 2D array
            'reconstruction': reconstructed 2D array
            'error_map': absolute difference map
            'metrics': dict with rmse, psnr, ssim
            'solver_info': dict with method details
            'b': flattened sinogram
    """
    size = image.shape[0]
    if n_angles is None:
        n_angles = size
    n_detectors = int(size * 1.4)

    theta = np.linspace(0, 180, n_angles, endpoint=False)
    sino = radon(image, theta=theta, circle=False)
    sino_flat = sino.flatten()

    noise_added = False
    if noise_level > 0:
        if seed is not None:
            np.random.seed(seed)
        sigma = noise_level * np.std(sino_flat)
        sino_noisy = sino_flat + np.random.randn(*sino_flat.shape) * sigma
        sino = sino_noisy.reshape(sino.shape)
        noise_added = True
    else:
        sino_noisy = sino_flat

    recon = iradon(sino, theta=theta, filter_name=filter, circle=False)

    recon = np.clip(recon, 0, 1)

    error_map = np.abs(image - recon)

    from src.metrics import compute_metrics, ssim
    metrics = compute_metrics(image.flatten(), recon.flatten())
    metrics['ssim'] = ssim(image.flatten(), recon.flatten(), size)
    if noise_added:
        metrics['noise_level'] = noise_level
    metrics['residual'] = np.linalg.norm(sino_noisy - radon(recon, theta=theta, circle=False).flatten()) / (np.linalg.norm(sino_noisy) + 1e-10)

    info = {
        'method': 'fbp',
        'factorization': f'iradon ({filter})',
        'angles': n_angles,
        'detectors': n_detectors,
        'filter': filter,
        'noise_level': noise_level if noise_added else 0,
    }

    return {
        'phantom': image,
        'reconstruction': recon,
        'error_map': error_map,
        'metrics': metrics,
        'solver_info': info,
        'b': sino_noisy,
    }
