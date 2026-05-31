"""Forward projector for parallel-beam CT geometry."""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import Tuple


def build_system(
    size: int, 
    n_angles: int = None, 
    n_detectors: int = None
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    """
    Build sparse system matrix A and sinogram b for parallel-beam CT.
    
    Args:
        size: Phantom dimension (size × size)
        n_angles: Number of projection angles (default: size)
        n_detectors: Number of detector bins (default: int(size * 1.4))
        
    Returns:
        A: Sparse system matrix (m × n) in CSR format
        b: Sinogram vector (m × 1)
        x_true: Ground truth flattened image (n × 1)
    """
    from .phantom import shepp_logan
    
    if n_angles is None:
        n_angles = size
    if n_detectors is None:
        n_detectors = int(size * 1.4)
    
    phantom = shepp_logan(size)
    x_true = phantom.flatten()
    
    n_pixels = size * size
    n_measurements = n_angles * n_detectors
    
    A = lil_matrix((n_measurements, n_pixels), dtype=np.float32)
    b = np.zeros(n_measurements, dtype=np.float32)
    
    angles = np.linspace(0, 180, n_angles, endpoint=False)
    detector_positions = np.linspace(-size / 1.4, size / 1.4, n_detectors)
    
    center = size / 2.0
    
    for angle_idx, angle in enumerate(angles):
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        for det_idx, det_pos in enumerate(detector_positions):
            row_idx = angle_idx * n_detectors + det_idx
            
            # Ray origin and direction
            ray_x = det_pos * (-sin_t) + center
            ray_y = det_pos * cos_t + center
            dir_x, dir_y = cos_t, sin_t
            
            # Trace ray through grid
            t_min, t_max = -size * 2, size * 2
            
            for i in range(size):
                for j in range(size):
                    # Pixel boundaries
                    x_min, x_max = j, j + 1
                    y_min, y_max = i, i + 1
                    
                    # Ray-box intersection
                    if abs(dir_x) > 1e-10:
                        t1 = (x_min - ray_x) / dir_x
                        t2 = (x_max - ray_x) / dir_x
                        t_x_min, t_x_max = min(t1, t2), max(t1, t2)
                    else:
                        t_x_min, t_x_max = -np.inf, np.inf
                    
                    if abs(dir_y) > 1e-10:
                        t1 = (y_min - ray_y) / dir_y
                        t2 = (y_max - ray_y) / dir_y
                        t_y_min, t_y_max = min(t1, t2), max(t1, t2)
                    else:
                        t_y_min, t_y_max = -np.inf, np.inf
                    
                    t_enter = max(t_x_min, t_y_min, t_min)
                    t_exit = min(t_x_max, t_y_max, t_max)
                    
                    if t_enter < t_exit:
                        length = t_exit - t_enter
                        if length > 1e-6:
                            pixel_idx = i * size + j
                            A[row_idx, pixel_idx] = length
                            b[row_idx] += length * phantom[i, j]
    
    return A.tocsr(), b, x_true


def get_sparsity(A: csr_matrix) -> float:
    """Calculate sparsity percentage of matrix A."""
    return 1.0 - A.nnz / (A.shape[0] * A.shape[1])
