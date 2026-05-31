"""Main CT reconstruction pipeline using LU decomposition."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from src.phantom import shepp_logan
from src.projector import build_system, get_sparsity
from src.lud_solver import solve_lu, iterative_refinement
from src.metrics import compute_metrics, ssim

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def reconstruct(size: int = 32, use_refinement: bool = False, method: str = 'auto'):
    """
    Full CT reconstruction pipeline.
    
    Args:
        size: Phantom dimension
        use_refinement: Apply iterative refinement
        method: Solver method ('auto', 'sparse', 'dense')
        
    Returns:
        Dictionary with reconstruction results
    """
    print(f"=== CT RECONSTRUCTION (size={size}) ===\n")
    
    # Step 1: Generate phantom and forward model
    print("1. Building forward model...")
    A, b, x_true = build_system(size)
    phantom = x_true.reshape(size, size)
    
    print(f"   ✓ A: {A.shape}, sparsity: {get_sparsity(A):.2%}")
    print(f"   ✓ b: {b.shape}")
    print(f"   ✓ Memory: {A.data.nbytes / 1024**2:.2f} MB")
    
    # Step 2: Solve Ax = b
    print(f"\n2. Solving Ax = b using LU decomposition...")
    x_rec, info = solve_lu(A, b, method=method)
    
    print(f"   ✓ Method: {info['method']} -> {info['factorization']}")
    print(f"   ✓ Initial residual: {info['residual']:.2e}")
    
    # Step 3: Optional refinement
    if use_refinement:
        print(f"\n3. Applying iterative refinement...")
        x_rec, ref_info = iterative_refinement(A, b, x_rec, max_iter=3)
        print(f"   ✓ Iterations: {ref_info['iterations']}")
        print(f"   ✓ Final residual: {ref_info['final_residual']:.2e}")
        info['residual'] = ref_info['final_residual']
    
    # Step 4: Compute metrics
    print(f"\n{'4' if use_refinement else '3'}. Computing quality metrics...")
    metrics = compute_metrics(x_true, x_rec, A, b)
    ssim_val = ssim(x_true, x_rec, size)
    metrics['ssim'] = ssim_val
    
    print(f"   ✓ RMSE: {metrics['rmse']:.4f}")
    print(f"   ✓ PSNR: {metrics['psnr']:.2f} dB")
    print(f"   ✓ SSIM: {metrics['ssim']:.4f}")
    print(f"   ✓ Relative error: {metrics['relative_error']:.4f}")
    print(f"   ✓ Residual: {metrics['residual']:.2e}")
    
    # Step 5: Visualize
    print(f"\n{'5' if use_refinement else '4'}. Generating visualizations...")
    reconstruction = x_rec.reshape(size, size)
    error_map = np.abs(phantom - reconstruction)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Ground truth
    im0 = axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth Phantom')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Reconstruction
    im1 = axes[0, 1].imshow(reconstruction, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'LU Reconstruction\nRMSE={metrics["rmse"]:.4f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Error map
    im2 = axes[1, 0].imshow(error_map, cmap='hot', vmin=0, vmax=error_map.max())
    axes[1, 0].set_title(f'Absolute Error\nMax={error_map.max():.4f}')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Sinogram
    sinogram = b.reshape(-1, int(size * 1.4))
    im3 = axes[1, 1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1, 1].set_title('Sinogram (Projection Data)')
    axes[1, 1].set_xlabel('Detector')
    axes[1, 1].set_ylabel('Angle')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    filename = 'phase3_reconstruction.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved: {filename}")
    
    print("\n=== RECONSTRUCTION COMPLETE ===")
    
    # Success criteria check
    print("\n=== SUCCESS CRITERIA ===")
    print(f"✓ RMSE < 0.05: {metrics['rmse']:.4f} {'✓' if metrics['rmse'] < 0.05 else '✗'}")
    print(f"✓ PSNR > 25 dB: {metrics['psnr']:.2f} {'✓' if metrics['psnr'] > 25 else '✗'}")
    print(f"✓ Residual < 1e-3: {metrics['residual']:.2e} {'✓' if metrics['residual'] < 1e-3 else '✗'}")
    
    return {
        'phantom': phantom,
        'reconstruction': reconstruction,
        'error_map': error_map,
        'metrics': metrics,
        'solver_info': info
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CT reconstruction using LU decomposition')
    parser.add_argument('--size', type=int, default=32, help='Phantom size (default: 32)')
    parser.add_argument('--refine', action='store_true', help='Use iterative refinement')
    parser.add_argument('--method', default='auto', choices=['auto', 'sparse', 'dense'],
                        help='Solver method (default: auto)')
    
    args = parser.parse_args()
    
    results = reconstruct(size=args.size, use_refinement=args.refine, method=args.method)
