"""3D Shepp-Logan phantom with evolving anatomy across slices."""

import numpy as np


def _slice_at(size: int, t: float) -> np.ndarray:
    """Generate a single Shepp-Logan slice at evolution parameter t in [0, 1].

    As t varies, tumors grow/shrink and features appear/disappear,
    creating a realistic 3D volume when stacked.
    """
    phantom = np.zeros((size, size))

    # Ellipses: (cx, cy, rx, ry, rot_deg, intensity)
    # Base skull (same across all slices)
    base = [
        (0.0, 0.0, 0.69, 0.92, 0, 1.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.8),
        (0.22, 0.0, 0.11, 0.31, -18, -0.2),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.2),
    ]

    # Tumor — grows then shrinks across slices
    tumor_cx = 0.0
    tumor_cy = 0.35 - 0.2 * (t - 0.5)
    tumor_rx = 0.21 + 0.08 * np.sin(t * np.pi)
    tumor_ry = 0.25 + 0.08 * np.sin(t * np.pi)
    tumor = (tumor_cx, tumor_cy, tumor_rx, tumor_ry, 0, 0.15)

    # Small nodule — appears in the middle third
    nodule_intensity = 0.1 if 0.3 < t < 0.7 else 0.0
    nodule = (0.0, 0.0, 0.046, 0.046, 0, nodule_intensity)

    # Bottom features — drift slightly
    bottom_left = (-0.08 + 0.04 * np.sin(t * 2 * np.pi), -0.605,
                   0.046, 0.023, 0, 0.1)
    bottom_center = (0.0, -0.605, 0.023, 0.023, 0, 0.1)
    bottom_right = (0.06 - 0.04 * np.cos(t * np.pi), -0.605,
                    0.023, 0.046, 0, 0.1)

    ellipses = base + [tumor, nodule, bottom_left, bottom_center, bottom_right]

    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]

    for cx, cy, rx, ry, angle, intensity in ellipses:
        if intensity == 0:
            continue
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = (x - cx) * cos_t + (y - cy) * sin_t
        y_rot = -(x - cx) * sin_t + (y - cy) * cos_t
        mask = (x_rot / rx) ** 2 + (y_rot / ry) ** 2 <= 1
        phantom[mask] += intensity

    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min())
    return phantom


def shepp_logan_3d(depth: int, size: int = 32) -> np.ndarray:
    """Generate a 3D Shepp-Logan volume with evolving anatomy.

    Args:
        depth: Number of slices along the Z-axis
        size: Height/width of each slice (size x size)

    Returns:
        3D array of shape (depth, size, size) with values in [0, 1]
    """
    volume = np.zeros((depth, size, size))
    for z in range(depth):
        t = z / (depth - 1) if depth > 1 else 0
        volume[z] = _slice_at(size, t)
    return volume
