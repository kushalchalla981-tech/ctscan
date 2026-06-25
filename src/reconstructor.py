"""Main CT reconstruction pipeline using LU decomposition."""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from src.projector import build_system, get_sparsity
from src.lud_solver import solve_lu, iterative_refinement
from src.metrics import compute_metrics, ssim
from src.noise import add_gaussian_noise
from src.fbp_solver import fbp_reconstruct

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def reconstruct(size: int = 32, use_refinement: bool = False, method: str = 'auto',
                input_image: str = None, compare_path: str = None,
                save_metrics_path: str = None,
                noise_level: float = 0.0, regularization: float = None,
                filter: str = 'ramp'):
    """
    Full CT reconstruction pipeline.

    Args:
        size: Phantom dimension
        use_refinement: Apply iterative refinement
        method: Solver method ('auto', 'sparse', 'dense')
        input_image: Path to DICOM or raster image to use as ground truth
        compare_path: Save side-by-side comparison plot to this path
        save_metrics_path: Save metrics JSON to this path
        noise_level: Relative Gaussian noise added to sinogram (0 = no noise)
        regularization: Tikhonov regularization strength (None = no regularization)

    Returns:
        Dictionary with reconstruction results
    """
    if method == 'fbp':
        return _reconstruct_fbp(size, input_image, noise_level, filter)

    source_label = 'Shepp-Logan Phantom'
    if input_image:
        from src.loader import load_image
        x_true = load_image(input_image, size).flatten()
        source_label = input_image
    else:
        from src.phantom import shepp_logan
        A, b, x_true = build_system(size)
        if noise_level > 0:
            b = add_gaussian_noise(b, noise_level)
            print(f"   Noise added: {noise_level*100:.1f}% Gaussian")

    phantom = x_true.reshape(size, size)

    print(f"=== CT RECONSTRUCTION (size={size}) ===\n")

    if input_image:
        print(f"Input image: {source_label}")
        print(f"This image is being used as the 'ground truth' (phantom).")
        print(f"We simulate X-rays passing through it from many angles,")
        print(f"then attempt to reconstruct the original from those projections.\n")
    else:
        print(f"Phantom: {source_label}")

    if not input_image:
        print(f"\n1. Building forward model...")
        print(f"   A: {A.shape}, sparsity: {get_sparsity(A):.2%}")
        print(f"   b: {b.shape}")
        print(f"   Memory: {A.data.nbytes / 1024**2:.2f} MB")
    else:
        print(f"\n1. Building forward model from input image...")
        A, b, _ = build_system(size)
        b = A @ x_true
        if noise_level > 0:
            b = add_gaussian_noise(b, noise_level)
            print(f"   Noise added: {noise_level*100:.1f}% Gaussian")
        print(f"   A: {A.shape}, sparsity: {get_sparsity(A):.2%}")
        print(f"   b: {b.shape}")

    print(f"\n2. Solving Ax = b...")
    reg_label = f", regularization={regularization}" if regularization else ""
    x_rec, info = solve_lu(A, b, method=method, regularization=regularization)
    print(f"   Method: {info['method']} -> {info['factorization']}{reg_label}")
    print(f"   Initial residual: {info['residual']:.2e}")

    if use_refinement:
        print(f"\n3. Applying iterative refinement...")
        x_rec, ref_info = iterative_refinement(A, b, x_rec, max_iter=3)
        print(f"   Iterations: {ref_info['iterations']}")
        print(f"   Final residual: {ref_info['final_residual']:.2e}")
        info['residual'] = ref_info['final_residual']

    step = 4 if use_refinement else 3
    print(f"\n{step}. Computing quality metrics...")
    metrics = compute_metrics(x_true, x_rec, A, b)
    ssim_val = ssim(x_true, x_rec, size)
    metrics['ssim'] = ssim_val
    metrics['source'] = source_label
    if noise_level > 0:
        metrics['noise_level'] = noise_level
    if regularization:
        metrics['regularization'] = regularization
    if use_refinement:
        metrics['refinement_iterations'] = ref_info['iterations']
        metrics['refinement_final_residual'] = ref_info['final_residual']

    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   Relative error: {metrics['relative_error']:.4f}")
    print(f"   Residual: {metrics['residual']:.2e}")

    reconstruction = x_rec.reshape(size, size)
    error_map = np.abs(phantom - reconstruction)

    if compare_path:
        _save_comparison(phantom, reconstruction, error_map, b, metrics, size, compare_path)

    if save_metrics_path:
        serializable = {k: float(v) if isinstance(v, (np.floating,)) else v
                        for k, v in metrics.items()}
        with open(save_metrics_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"   Metrics saved: {save_metrics_path}")

    print("\n=== RECONSTRUCTION COMPLETE ===")
    if not input_image:
        print(f"\nSuccess Criteria:")
        print(f"RMSE < 0.05: {metrics['rmse']:.4f} {'OK' if metrics['rmse'] < 0.05 else 'FAIL'}")
        print(f"PSNR > 25 dB: {metrics['psnr']:.2f} {'OK' if metrics['psnr'] > 25 else 'FAIL'}")
        print(f"Residual < 1e-3: {metrics['residual']:.2e} {'OK' if metrics['residual'] < 1e-3 else 'FAIL'}")

    return {
        'phantom': phantom,
        'reconstruction': reconstruction,
        'error_map': error_map,
        'metrics': metrics,
        'solver_info': info,
        'b': b,
    }


def _save_comparison(phantom, reconstruction, error_map, b, metrics, size, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(reconstruction, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Reconstruction\nRMSE={metrics["rmse"]:.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
    axes[1, 0].set_title(f'Absolute Error\nMax={error_map.max():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    sinogram = b.reshape(-1, int(size * 1.4))
    im3 = axes[1, 1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Sinogram (Projection Data)')
    axes[1, 1].set_xlabel('Detector')
    axes[1, 1].set_ylabel('Angle')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Comparison saved: {path}")


def _reconstruct_fbp(size, input_image, noise_level, filter='ramp'):
    """FBP reconstruction path: radon forward + iradon inverse."""
    from src.phantom import shepp_logan
    from src.loader import load_image
    from src.metrics import compute_metrics, ssim

    if input_image:
        x_true = load_image(input_image, size)
        source_label = input_image
    else:
        x_true = shepp_logan(size)
        source_label = 'Shepp-Logan Phantom'

    print(f"=== FBP RECONSTRUCTION (size={size}) ===\n")
    print(f"Phantom: {source_label}")
    print(f"\n1. Forward Radon transform + inverse FBP (filter={filter})...")

    result = fbp_reconstruct(x_true, n_angles=size, filter=filter, noise_level=noise_level)

    metrics = result['metrics']
    print(f"   Method: FBP -> iradon ({result['solver_info']['filter']})")
    if noise_level > 0:
        print(f"   Noise added: {noise_level*100:.1f}% Gaussian")
    print(f"\n2. Computing quality metrics...")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")

    result['metrics']['source'] = source_label
    result['metrics']['filter'] = filter
    if noise_level > 0:
        result['metrics']['noise_level'] = noise_level

    print("\n=== FBP RECONSTRUCTION COMPLETE ===")
    return result


if __name__ == "__main__":
    print("Use 'python main.py reconstruct' instead.")
    print("See 'python main.py --help' for details.")
