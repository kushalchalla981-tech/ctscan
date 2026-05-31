"""Unit tests for Phase 2 LU solver."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye as sparse_eye
from src.lud_solver import solve_lu, iterative_refinement, _dense_lu_solve


def test_dense_lu_identity():
    """Test dense LU on identity matrix."""
    A = np.eye(5)
    b = np.array([1, 2, 3, 4, 5])
    x = _dense_lu_solve(A, b)
    np.testing.assert_allclose(x, b, rtol=1e-10)


def test_dense_lu_random():
    """Test dense LU on random well-conditioned matrix."""
    np.random.seed(42)
    n = 20
    A = np.random.randn(n, n) + 5 * np.eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x = _dense_lu_solve(A, b)
    np.testing.assert_allclose(x, x_true, rtol=1e-8)


def test_solve_lu_dense():
    """Test solve_lu with dense method."""
    np.random.seed(42)
    A = np.random.randn(10, 10) + 3 * np.eye(10)
    x_true = np.random.randn(10)
    b = A @ x_true
    
    x, info = solve_lu(A, b, method='dense')
    assert info['residual'] < 1e-10
    np.testing.assert_allclose(x, x_true, rtol=1e-8)


def test_solve_lu_sparse():
    """Test solve_lu with sparse method."""
    np.random.seed(42)
    n = 50
    A = csr_matrix(np.random.randn(n, n)) + 3 * sparse_eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x, info = solve_lu(A, b, method='sparse')
    assert info['residual'] < 1e-8
    np.testing.assert_allclose(x, x_true, rtol=1e-6)


def test_solve_lu_auto():
    """Test automatic method selection."""
    # Small dense
    A_small = np.random.randn(50, 50) + 2 * np.eye(50)
    b_small = np.random.randn(50)
    x, info = solve_lu(A_small, b_small, method='auto')
    assert info['method'] in ['dense', 'sparse']
    
    # Large sparse
    A_large = csr_matrix(np.random.randn(150, 150)) + 2 * sparse_eye(150)
    b_large = np.random.randn(150)
    x, info = solve_lu(A_large, b_large, method='auto')
    assert info['method'] == 'sparse'


def test_iterative_refinement_improves():
    """Test that iterative refinement improves solution."""
    np.random.seed(42)
    n = 30
    A = np.random.randn(n, n) + 4 * np.eye(n)
    x_true = np.random.randn(n)
    b = A @ x_true
    
    x0, info0 = solve_lu(A, b)
    x_ref, info_ref = iterative_refinement(A, b, x0, max_iter=2)
    
    assert info_ref['final_residual'] <= info0['residual']


def test_singular_matrix_raises():
    """Test that singular matrix raises error."""
    A = np.array([[1, 2], [2, 4]], dtype=float)  # Singular
    b = np.array([1, 2], dtype=float)
    
    with pytest.raises(np.linalg.LinAlgError):
        _dense_lu_solve(A, b)
