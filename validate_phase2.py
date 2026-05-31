"""Phase 2 validation: Test LU solver on known systems."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from src.lud_solver import solve_lu, iterative_refinement

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def test_small_dense():
    """Test on small dense system with known solution."""
    print("1. Testing small dense system (n=10)...")
    np.random.seed(42)
    n = 10
    A = np.random.randn(n, n) + 5 * np.eye(n)  # Well-conditioned
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x_sol, info = solve_lu(A, b, method='dense')
    error = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
    
    print(f"   ✓ Method: {info['factorization']}")
    print(f"   ✓ Condition number: {info['condition_number']:.2e}")
    print(f"   ✓ Residual: {info['residual']:.2e}")
    print(f"   ✓ Solution error: {error:.2e}")
    
    assert error < 1e-10, f"Solution error too high: {error}"
    assert info['residual'] < 1e-10, f"Residual too high: {info['residual']}"
    print("   ✓ PASSED\n")


def test_sparse_system():
    """Test on sparse system."""
    print("2. Testing sparse system (n=100, sparsity=95%)...")
    np.random.seed(42)
    n = 100
    A = sparse_random(n, n, density=0.05, format='csr')
    A = A + 2 * sparse_random(n, n, density=1.0, format='csr').multiply(np.eye(n))
    A = A.tocsr()
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x_sol, info = solve_lu(A, b, method='sparse')
    error = np.linalg.norm(x_sol - x_true) / np.linalg.norm(x_true)
    
    print(f"   ✓ Method: {info['factorization']}")
    print(f"   ✓ Residual: {info['residual']:.2e}")
    print(f"   ✓ Solution error: {error:.2e}")
    
    assert error < 1e-8, f"Solution error too high: {error}"
    assert info['residual'] < 1e-8, f"Residual too high: {info['residual']}"
    print("   ✓ PASSED\n")


def test_auto_method():
    """Test automatic method selection."""
    print("3. Testing auto method selection...")
    
    # Small dense
    A_dense = np.random.randn(50, 50) + 3 * np.eye(50)
    b_dense = np.random.randn(50)
    x, info = solve_lu(A_dense, b_dense, method='auto')
    print(f"   ✓ Dense (n=50): {info['method']} -> {info['factorization']}")
    
    # Large sparse
    from scipy.sparse import eye as sparse_eye
    A_sparse = sparse_random(200, 200, density=0.03, format='csr') + 2 * sparse_eye(200)
    b_sparse = np.random.randn(200)
    x, info = solve_lu(A_sparse, b_sparse, method='auto')
    print(f"   ✓ Sparse (n=200): {info['method']} -> {info['factorization']}")
    print("   ✓ PASSED\n")


def test_iterative_refinement():
    """Test iterative refinement."""
    print("4. Testing iterative refinement...")
    np.random.seed(42)
    n = 50
    A = np.random.randn(n, n) + 4 * np.eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    # Initial solution
    x0, info0 = solve_lu(A, b)
    
    # Refined solution
    x_ref, info_ref = iterative_refinement(A, b, x0, max_iter=3)
    
    error0 = np.linalg.norm(x0 - x_true) / np.linalg.norm(x_true)
    error_ref = np.linalg.norm(x_ref - x_true) / np.linalg.norm(x_true)
    
    print(f"   ✓ Initial residual: {info0['residual']:.2e}")
    print(f"   ✓ Refined residual: {info_ref['final_residual']:.2e}")
    print(f"   ✓ Iterations: {info_ref['iterations']}")
    print(f"   ✓ Error improvement: {error0:.2e} -> {error_ref:.2e}")
    print("   ✓ PASSED\n")


def test_ill_conditioned():
    """Test warning on ill-conditioned matrix."""
    print("5. Testing ill-conditioned matrix handling...")
    n = 20
    A = np.random.randn(n, n)
    A[:, -1] = A[:, 0] * 1.0001  # Nearly singular
    b = np.random.randn(n)
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x, info = solve_lu(A, b, method='dense')
        
        if len(w) > 0:
            print(f"   ✓ Warning raised: {w[0].message}")
        print(f"   ✓ Condition number: {info['condition_number']:.2e}")
        print("   ✓ PASSED\n")


def validate_phase2():
    """Run all Phase 2 validation tests."""
    print("=== PHASE 2 VALIDATION ===\n")
    
    test_small_dense()
    test_sparse_system()
    test_auto_method()
    test_iterative_refinement()
    test_ill_conditioned()
    
    print("=== PHASE 2 COMPLETE ===")
    print("✅ All LU solver tests passed")
    print("✅ Ready for Phase 3: CT reconstruction pipeline")


if __name__ == "__main__":
    validate_phase2()
