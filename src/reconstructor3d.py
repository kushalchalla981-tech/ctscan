"""3D CT reconstruction by stacking 2D slice reconstructions."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from src.phantom3d import shepp_logan_3d
from src.projector import build_system, get_sparsity
from src.lud_solver import solve_lu
from src.metrics import compute_metrics, ssim
from src.noise import add_gaussian_noise

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def reconstruct_3d(
    depth: int = 16,
    size: int = 32,
    method: str = 'auto',
    noise_level: float = 0.0,
    regularization: float = None,
    filter: str = 'ramp',
    compare_path: str = None,
) -> dict:
    """
    Reconstruct a 3D volume slice-by-slice.

    Each Z-slice is independently reconstructed using the specified
    2D method ('auto', 'sparse', 'dense', or 'fbp'), then stacked.

    Args:
        depth: Number of slices along Z
        size: Slice dimension (size x size)
        method: Reconstruction method per slice
        noise_level: Gaussian noise added to sinogram per slice
        regularization: Tikhonov regularization (LSQR methods only)
        filter: FBP filter ('ramp', 'shepp-logan', ...)
        compare_path: Save visualization to this path

    Returns:
        dict with volume_true, volume_rec, volume_error, metrics, ...
    """
    print(f"=== 3D CT RECONSTRUCTION ({depth} x {size}x{size}) ===\n")
    print(f"Method: {method.upper()}\n")

    volume_true = shepp_logan_3d(depth, size)
    volume_rec = np.zeros_like(volume_true)
    volume_error = np.zeros_like(volume_true)
    all_metrics = []

    if method == 'fbp':
        from src.fbp_solver import fbp_reconstruct
        print(f"1. Reconstructing {depth} slices via FBP (filter={filter})...")
        for z in range(depth):
            result = fbp_reconstruct(
                volume_true[z], n_angles=size,
                filter=filter, noise_level=noise_level,
            )
            volume_rec[z] = result['reconstruction']
            volume_error[z] = result['error_map']
            all_metrics.append(result['metrics'])
            _print_progress(z, depth)

    else:
        print(f"1. Building system matrix A (size={size})...")
        A, _, _ = build_system(size)
        print(f"   A: {A.shape}, sparsity: {get_sparsity(A):.2%}")
        print(f"2. Reconstructing {depth} slices via {method.upper()}...")
        for z in range(depth):
            x_true = volume_true[z].flatten()
            b = A @ x_true
            if noise_level > 0:
                b = add_gaussian_noise(b, noise_level)
            x_rec, info = solve_lu(A, b, method=method, regularization=regularization)
            rec_slice = x_rec.reshape(size, size)
            volume_rec[z] = rec_slice
            volume_error[z] = np.abs(volume_true[z] - rec_slice)
            m = compute_metrics(x_true, x_rec)
            m['ssim'] = ssim(x_true, x_rec, size)
            m['slice'] = z
            all_metrics.append(m)
            _print_progress(z, depth)

    avg_rmse = np.mean([m['rmse'] for m in all_metrics])
    avg_psnr = np.mean([m['psnr'] for m in all_metrics])
    avg_ssim = np.mean([m['ssim'] for m in all_metrics])

    summary_metrics = {
        'rmse': avg_rmse,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'depth': depth,
        'size': size,
        'method': method,
        'noise_level': noise_level,
        'regularization': regularization,
        'filter': filter,
        'per_slice': all_metrics,
    }

    print(f"\n3. Volume quality metrics:")
    print(f"   Avg RMSE: {avg_rmse:.4f}")
    print(f"   Avg PSNR: {avg_psnr:.2f} dB")
    print(f"   Avg SSIM: {avg_ssim:.4f}")

    if compare_path:
        _save_volume_comparison(
            volume_true, volume_rec, volume_error,
            summary_metrics, compare_path,
        )

    print("\n=== 3D RECONSTRUCTION COMPLETE ===")
    return {
        'volume_true': volume_true,
        'volume_rec': volume_rec,
        'volume_error': volume_error,
        'metrics': summary_metrics,
    }


def _print_progress(current, total):
    if total <= 1:
        return
    pct = (current + 1) / total * 100
    bar_len = 30
    filled = int(bar_len * (current + 1) / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f'\r   Slice {current + 1}/{total}  [{bar}] {pct:.0f}%', end='')
    if current == total - 1:
        print()


def _save_volume_comparison(vol_true, vol_rec, vol_error, metrics, path):
    depth, size = vol_true.shape[0], vol_true.shape[1]

    slice_indices = _pick_slices(depth)

    n_slices = len(slice_indices)
    fig, axes = plt.subplots(4, n_slices, figsize=(3 * n_slices, 10))

    for col, z in enumerate(slice_indices):
        axes[0, col].imshow(vol_true[z], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'Truth Z={z}')
        axes[0, col].axis('off')

        axes[1, col].imshow(vol_rec[z], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f'Recon Z={z}')
        axes[1, col].axis('off')

        axes[2, col].imshow(vol_error[z], cmap='hot')
        axes[2, col].set_title(f'Error Z={z}')
        axes[2, col].axis('off')

        axes[3, col].imshow(vol_true[z] - vol_rec[z], cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[3, col].set_title(f'Diff Z={z}')
        axes[3, col].axis('off')

    for row, label in enumerate(['Ground Truth', 'Reconstruction', 'Abs Error', 'Signed Diff']):
        axes[row, 0].set_ylabel(label, fontsize=10)

    fig.suptitle(
        f'3D Reconstruction — {metrics["method"].upper()}  |  '
        f'RMSE={metrics["rmse"]:.4f}  PSNR={metrics["psnr"]:.1f}dB  SSIM={metrics["ssim"]:.4f}',
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Comparison saved: {path}")


def _pick_slices(depth):
    if depth <= 5:
        return list(range(depth))
    picks = set()
    picks.add(0)
    picks.add(depth - 1)
    for frac in [0.25, 0.5, 0.75]:
        picks.add(int(depth * frac))
    return sorted(picks)
