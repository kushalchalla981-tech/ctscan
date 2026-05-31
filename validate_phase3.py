"""Phase 3 validation: Test full reconstruction pipeline."""

import sys
import numpy as np
from src.reconstructor import reconstruct

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def validate_phase3():
    """Validate Phase 3 reconstruction pipeline."""
    print("=== PHASE 3 VALIDATION ===\n")
    
    # Test 1: Basic reconstruction (32x32)
    print("Test 1: Basic reconstruction (32×32)")
    print("-" * 50)
    results = reconstruct(size=32, use_refinement=False, method='auto')
    
    metrics = results['metrics']
    
    # Validate success criteria
    assert metrics['rmse'] < 0.05, f"RMSE too high: {metrics['rmse']}"
    assert metrics['psnr'] > 25, f"PSNR too low: {metrics['psnr']}"
    assert metrics['residual'] < 1e-3, f"Residual too high: {metrics['residual']}"
    
    print("\n✅ Test 1 PASSED\n")
    
    # Test 2: With iterative refinement
    print("\nTest 2: Reconstruction with iterative refinement")
    print("-" * 50)
    results_ref = reconstruct(size=32, use_refinement=True, method='auto')
    
    metrics_ref = results_ref['metrics']
    
    # Refinement should maintain or improve quality
    assert metrics_ref['residual'] <= metrics['residual'] * 1.1, "Refinement degraded solution"
    
    print("\n✅ Test 2 PASSED\n")
    
    print("\n" + "=" * 50)
    print("=== PHASE 3 COMPLETE ===")
    print("✅ All reconstruction tests passed")
    print("✅ Success criteria met:")
    print(f"   • RMSE: {metrics['rmse']:.4f} < 0.05")
    print(f"   • PSNR: {metrics['psnr']:.2f} dB > 25")
    print(f"   • Residual: {metrics['residual']:.2e} < 1e-3")
    print("\n✅ Ready for Phase 4: Noise robustness testing")


if __name__ == "__main__":
    validate_phase3()
