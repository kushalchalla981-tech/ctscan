"""Shepp-Logan phantom generation for CT reconstruction."""

import numpy as np
from typing import Tuple


def shepp_logan(size: int) -> np.ndarray:
    """
    Generate modified Shepp-Logan phantom.
    
    Args:
        size: Image dimension (size × size pixels)
        
    Returns:
        2D array of pixel intensities [0, 1]
    """
    phantom = np.zeros((size, size))
    
    # Ellipse parameters: (x_center, y_center, x_radius, y_radius, rotation_deg, intensity)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),      # Outer skull
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),  # Inner skull
        (0.22, 0.0, 0.11, 0.31, -18, -0.2),  # Right hemisphere
        (-0.22, 0.0, 0.16, 0.41, 18, -0.2),  # Left hemisphere
        (0.0, 0.35, 0.21, 0.25, 0, 0.1),     # Tumor/feature
        (0.0, 0.1, 0.046, 0.046, 0, 0.1),    # Small feature
        (0.0, -0.1, 0.046, 0.046, 0, 0.1),   # Small feature
        (-0.08, -0.605, 0.046, 0.023, 0, 0.1),  # Bottom left
        (0.0, -0.605, 0.023, 0.023, 0, 0.1),    # Bottom center
        (0.06, -0.605, 0.023, 0.046, 0, 0.1),   # Bottom right
    ]
    
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    
    for cx, cy, rx, ry, angle, intensity in ellipses:
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        
        x_rot = (x - cx) * cos_t + (y - cy) * sin_t
        y_rot = -(x - cx) * sin_t + (y - cy) * cos_t
        
        mask = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1
        phantom[mask] += intensity
    
    # Normalize to [0, 1]
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    return phantom
