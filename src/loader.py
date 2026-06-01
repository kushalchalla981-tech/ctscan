"""Image loader supporting DICOM and raster formats."""

import numpy as np
from pathlib import Path


def load_image(path: str, size: int = 32) -> np.ndarray:
    """
    Load an image from path and return as normalized [0,1] 2D array.

    Auto-detects DICOM vs raster format. For DICOM, applies window
    center/width and rescale slope/intercept if present in metadata.

    Args:
        path: Path to image file (.dcm, .png, .jpg, .tif, etc.)
        size: Output image dimension (size x size)

    Returns:
        2D float64 array normalized to [0, 1]

    Raises:
        FileNotFoundError: If path does not exist
        ValueError: If image cannot be read
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    data = _try_load_dicom(path)
    if data is None:
        data = _try_load_raster(path)
    if data is None:
        raise ValueError(f"Could not read image: {path}")

    data = _resize(data, size)

    data = data.astype(np.float64)
    lo, hi = data.min(), data.max()
    if hi > lo:
        data = (data - lo) / (hi - lo)

    return data


def _try_load_dicom(path: Path):
    """Try to load as DICOM. Returns 2D array or None."""
    try:
        import pydicom
        ds = pydicom.dcmread(str(path), force=True)
        if hasattr(ds, 'pixel_array') and ds.pixel_array.size > 0:
            arr = ds.pixel_array.astype(np.float64)

            # Apply RescaleSlope/Intercept (convert to HU)
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
            if slope != 1 or intercept != 0:
                arr = arr * slope + intercept

            # Apply Window Center/Width for soft-tissue windowing
            wc = getattr(ds, 'WindowCenter', None)
            ww = getattr(ds, 'WindowWidth', None)
            if wc is not None and ww is not None:
                wc_val = wc[0] if isinstance(wc, (list, tuple, pydicom.multival.MultiValue)) else wc
                ww_val = ww[0] if isinstance(ww, (list, tuple, pydicom.multival.MultiValue)) else ww
                lo = wc_val - ww_val / 2.0
                hi = wc_val + ww_val / 2.0
                arr = np.clip(arr, lo, hi)

            # Squeeze multi-frame or 3D down to single 2D slice
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            elif arr.ndim > 3:
                arr = arr[0, 0]

            return arr
    except Exception:
        pass
    return None


def _try_load_raster(path: Path):
    """Try to load as raster image (PNG, JPG, etc.). Returns 2D or None."""
    try:
        import skimage.io
        import skimage.color
        arr = skimage.io.imread(str(path))
        if arr.ndim == 3:
            arr = skimage.color.rgb2gray(arr)
        return arr.astype(np.float64)
    except Exception:
        pass
    return None


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize 2D array to (size x size)."""
    if arr.shape == (size, size):
        return arr
    from skimage.transform import resize
    return resize(arr, (size, size), anti_aliasing=True, preserve_range=True)
