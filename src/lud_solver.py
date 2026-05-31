"""LU decomposition solver with partial pivoting."""

import numpy as np
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import spsolve, splu, lsqr
from typing import Tuple, Optional
import warnings


def solve_lu(A, b, method='auto', check_condition=True, regularization=None) -> Tuple[np.ndarray, dict]:
    """
    Solve Ax = b using LU decomposition with partial pivoting.
    For rectangular A (overdetermined), uses least squares solver.
    
    Args:
        A: System matrix (dense or sparse)
        b: Right-hand side vector
        method: 'sparse', 'dense', or 'auto' (default)
        check_condition: Check condition number and warn if ill-conditioned
        regularization: Tikhonov regularization parameter (for overdetermined systems)
        
    Returns:
        x: Solution vector
        info: Dictionary with solver statistics
    """
    info = {'method': method, 'residual': 0.0, 'condition_number': None}
    
    # Handle rectangular matrices (overdetermined systems)
    is_rectangular = A.shape[0] != A.shape[1]
    if is_rectangular:
        info['system_type'] = 'overdetermined'
        
        # Use least squares solver for overdetermined systems
        if issparse(A):
            # Use LSQR for sparse least squares
            if regularization is None:
                regularization = 0.0
            result = lsqr(A, b, damp=np.sqrt(regularization) if regularization > 0 else 0.0)
            x = result[0]
            info['factorization'] = 'LSQR'
            info['lsqr_iterations'] = result[2]
            if regularization > 0:
                info['regularization'] = regularization
        else:
            # Use numpy's least squares for dense
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            info['factorization'] = 'numpy_lstsq'
            info['rank'] = rank
        
        # Compute residual
        residual_vec = A @ x - b
        info['residual'] = np.linalg.norm(residual_vec) / np.linalg.norm(b)
        
        return x, info
    
    # Square system - use LU decomposition
    info['system_type'] = 'square'
    
    # Auto-detect method
    if method == 'auto':
        if issparse(A):
            method = 'sparse' if A.shape[0] > 100 else 'dense'
        else:
            method = 'dense'
        info['method'] = method
    
    # Sparse solver
    if method == 'sparse':
        if not issparse(A):
            A = csr_matrix(A)
        
        try:
            lu = splu(A.tocsc())
            x = lu.solve(b)
            info['factorization'] = 'SuperLU'
        except Exception as e:
            warnings.warn(f"Sparse LU failed: {e}. Falling back to spsolve.")
            x = spsolve(A, b)
            info['factorization'] = 'spsolve'
    
    # Dense solver
    else:
        if issparse(A):
            A = A.toarray()
        
        if check_condition:
            try:
                cond = np.linalg.cond(A)
                info['condition_number'] = cond
                if cond > 1e10:
                    warnings.warn(f"Matrix is ill-conditioned (cond={cond:.2e}). Solution may be inaccurate.")
            except:
                pass
        
        try:
            x = _dense_lu_solve(A, b)
            info['factorization'] = 'dense_pivoted'
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Dense LU failed: {e}. Using numpy.linalg.solve.")
            x = np.linalg.solve(A, b)
            info['factorization'] = 'numpy_solve'
    
    # Compute residual
    if issparse(A):
        residual_vec = A @ x - b
    else:
        residual_vec = A @ x - b
    info['residual'] = np.linalg.norm(residual_vec) / np.linalg.norm(b)
    
    return x, info


def _dense_lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using dense LU with partial pivoting.
    Implements PA = LU factorization.
    """
    n = A.shape[0]
    A_work = A.astype(np.float64, copy=True)
    b_work = b.astype(np.float64, copy=True)
    P = np.arange(n)
    
    # LU factorization with partial pivoting
    for k in range(n - 1):
        # Find pivot
        pivot_row = k + np.argmax(np.abs(A_work[k:n, k]))
        
        if abs(A_work[pivot_row, k]) < 1e-14:
            raise np.linalg.LinAlgError(f"Matrix is singular at column {k}")
        
        # Swap rows
        if pivot_row != k:
            A_work[[k, pivot_row]] = A_work[[pivot_row, k]]
            b_work[[k, pivot_row]] = b_work[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
        
        # Eliminate
        A_work[k+1:n, k] /= A_work[k, k]
        A_work[k+1:n, k+1:n] -= np.outer(A_work[k+1:n, k], A_work[k, k+1:n])
    
    # Forward substitution (Ly = Pb)
    y = b_work.copy()
    for i in range(n):
        y[i] -= A_work[i, :i] @ y[:i]
    
    # Backward substitution (Ux = y)
    x = y.copy()
    for i in range(n - 1, -1, -1):
        if abs(A_work[i, i]) < 1e-14:
            raise np.linalg.LinAlgError(f"Matrix is singular at row {i}")
        x[i] = (x[i] - A_work[i, i+1:n] @ x[i+1:n]) / A_work[i, i]
    
    return x


def iterative_refinement(A, b, x0, max_iter=3) -> Tuple[np.ndarray, dict]:
    """
    Improve solution accuracy using iterative refinement.
    
    Args:
        A: System matrix
        b: Right-hand side
        x0: Initial solution
        max_iter: Maximum refinement iterations
        
    Returns:
        x: Refined solution
        info: Refinement statistics
    """
    x = x0.copy()
    residuals = []
    
    for i in range(max_iter):
        r = b - (A @ x)
        residual_norm = np.linalg.norm(r) / np.linalg.norm(b)
        residuals.append(residual_norm)
        
        if residual_norm < 1e-10:
            break
        
        # Solve for correction
        dx, _ = solve_lu(A, r, check_condition=False)
        x += dx
    
    info = {
        'iterations': len(residuals),
        'residuals': residuals,
        'final_residual': residuals[-1]
    }
    
    return x, info
