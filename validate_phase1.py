"""Phase 1 validation: Test phantom generation and forward projector."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from src.phantom import shepp_logan
from src.projector import build_system, get_sparsity

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def validate_phase1(size: int = 32):
    """Validate Phase 1 components."""
    print(f"=== PHASE 1 VALIDATION (size={size}) ===\n")
    
    # Generate phantom
    print("1. Generating Shepp-Logan phantom...")
    phantom = shepp_logan(size)
    print(f"   ✓ Phantom shape: {phantom.shape}")
    print(f"   ✓ Value range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    
    # Build system
    print("\n2. Building sparse system matrix A and sinogram b...")
    A, b, x_true = build_system(size)
    
    print(f"   ✓ A shape: {A.shape} (m={A.shape[0]}, n={A.shape[1]})")
    print(f"   ✓ A non-zeros: {A.nnz:,}")
    print(f"   ✓ A sparsity: {get_sparsity(A):.2%}")
    print(f"   ✓ b shape: {b.shape}")
    print(f"   ✓ b range: [{b.min():.3f}, {b.max():.3f}]")
    print(f"   ✓ x_true shape: {x_true.shape}")
    
    # Verify forward model
    print("\n3. Verifying forward model (Ax = b)...")
    b_computed = A @ x_true
    residual = np.linalg.norm(b - b_computed) / np.linalg.norm(b)
    print(f"   ✓ Residual ||Ax_true - b|| / ||b||: {residual:.2e}")
    
    if residual < 1e-6:
        print("   ✓ Forward model verified!")
    else:
        print(f"   ⚠ Warning: Residual higher than expected")
    
    # Visualize
    print("\n4. Generating visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(phantom, cmap='gray')
    axes[0].set_title('Shepp-Logan Phantom')
    axes[0].axis('off')
    
    sinogram = b.reshape(-1, int(size * 1.4))
    axes[1].imshow(sinogram, cmap='gray', aspect='auto')
    axes[1].set_title('Sinogram (b)')
    axes[1].set_xlabel('Detector')
    axes[1].set_ylabel('Angle')
    
    axes[2].spy(A[:500, :500], markersize=0.5)
    axes[2].set_title('A Sparsity Pattern (first 500×500)')
    
    plt.tight_layout()
    plt.savefig('phase1_validation.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: phase1_validation.png")
    
    print("\n=== PHASE 1 COMPLETE ===")
    print(f"Memory estimate: ~{A.data.nbytes / 1024**2:.1f} MB for A")
    
    return A, b, x_true, phantom


if __name__ == "__main__":
    A, b, x_true, phantom = validate_phase1(32)
    print("\n✅ Ready for Phase 2: LU solver implementation")
