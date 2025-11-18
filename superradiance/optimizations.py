"""
Optimization Functions for Superradiance Simulation
===================================================
This module contains the optimization-specific functions and classes.
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from numba import njit
from core import build_Gamma as build_Gamma_base

@njit
def compute_distances_numba(positions):
    """Fast distance computation using Numba JIT"""
    N = len(positions)
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                r = 0.0
                for k in range(len(positions[i])):
                    r += (positions[i][k] - positions[j][k])**2
                distances[i, j] = np.sqrt(r)
    return distances

def build_Gamma_optimized(positions, gamma0=1.0, k0=2*np.pi):
    """
    Optimized Gamma matrix construction using vectorized operations
    """
    N = len(positions)
    positions_array = np.array(positions, dtype=np.float64)
    
    # Use numba for distance calculation
    distances = compute_distances_numba(positions_array)
    
    # Initialize Gamma matrix
    Gamma = np.zeros((N, N))
    np.fill_diagonal(Gamma, gamma0)
    
    # Vectorized sinc calculation for off-diagonal elements
    mask = distances > 0
    kr_vals = k0 * distances[mask]
    sinc_vals = np.sinc(kr_vals / np.pi)
    Gamma[mask] = gamma0 * sinc_vals
    
    return 0.5 * (Gamma + Gamma.T)

def build_jump_ops_optimized(Gamma, lower_ops):
    """
    Optimized jump operator construction
    """
    vals, vecs = la.eigh(Gamma)
    ops = []
    
    # Only keep positive eigenvalues
    positive_mask = vals > 1e-12
    vals = vals[positive_mask]
    vecs = vecs[:, positive_mask]
    
    for eigval, eigvec in zip(vals, vecs.T):
        # Efficient linear combination
        cq = sum(vj * op for vj, op in zip(eigvec, lower_ops))
        ops.append(np.sqrt(eigval) * cq)
    
    return ops

class FastMatrixExponential:
    """
    Fast matrix exponential with caching and Taylor approximation
    
    This class provides optimized matrix exponential computation by:
    1. Using Taylor expansion for small time steps
    2. Caching frequently used exponentials
    3. Supporting sparse matrices for large systems
    """
    
    def __init__(self, Heff, max_cache_size=500, taylor_threshold=1e-5):
        self.Heff = Heff
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.taylor_threshold = taylor_threshold
        
        # Pre-compute identity and powers for Taylor expansion
        self.I = np.eye(Heff.shape[0], dtype=complex)
        
        # Check if we should use sparse matrices
        self.use_sparse = Heff.shape[0] > 32 and np.count_nonzero(Heff) / Heff.size < 0.1
        if self.use_sparse:
            self.Heff_sparse = csr_matrix(Heff)
        
        # Pre-compute powers for Taylor series
        if not self.use_sparse and Heff.shape[0] <= 128:  # Only for reasonable sizes
            self.Heff2 = Heff @ Heff
            self.Heff3 = self.Heff2 @ Heff
            self.use_precomputed = True
        else:
            self.use_precomputed = False
    
    def __call__(self, tau):
        """Compute matrix exponential exp(H_eff * tau)"""
        # For very small tau, use Taylor expansion
        if abs(tau) < self.taylor_threshold:
            return self._taylor_expansion(tau)
        
        # Check cache
        tau_key = round(tau, 8)
        if tau_key in self.cache:
            return self.cache[tau_key]
        
        # Compute exponential
        if self.use_sparse:
            # For sparse matrices, we typically use this with matrix-vector products
            # Return a function that can be applied to vectors
            return lambda v: expm_multiply(self.Heff_sparse * tau, v)
        else:
            exp_mat = la.expm(self.Heff * tau)
        
        # Cache result
        if len(self.cache) < self.max_cache_size:
            self.cache[tau_key] = exp_mat
        
        return exp_mat
    
    def _taylor_expansion(self, tau):
        """3rd order Taylor expansion for small tau"""
        if self.use_precomputed:
            Htau = self.Heff * tau
            Htau2 = self.Heff2 * (tau * tau)
            Htau3 = self.Heff3 * (tau * tau * tau)
            return self.I + Htau + 0.5 * Htau2 + (1.0/6.0) * Htau3
        else:
            Htau = self.Heff * tau
            return self.I + Htau + 0.5 * (Htau @ Htau)

class OptimizedEvolution:
    """
    Optimized quantum evolution class
    
    Handles efficient state evolution with:
    - Pre-computed rate operators
    - Sparse matrix support
    - Memory-efficient operations
    """
    
    def __init__(self, system_dict):
        """
        Initialize with system dictionary from core.setup_system()
        """
        self.H_eff = system_dict['H_eff']
        self.jump_ops = system_dict['jump_ops']
        self.I_op = system_dict['I_op']
        
        # Pre-compute rate operators
        self.rate_ops = [c.conj().T @ c for c in self.jump_ops]
        
        # Create fast matrix exponential
        self.fast_exp = FastMatrixExponential(self.H_eff)
        
        # Pre-allocate arrays
        self.rates = np.zeros(len(self.jump_ops))
    
    def compute_rates(self, psi):
        """Compute jump rates for given state"""
        for i, rate_op in enumerate(self.rate_ops):
            self.rates[i] = np.real(np.vdot(psi, rate_op @ psi))
        return self.rates
    
    def evolve(self, psi, tau):
        """Evolve state by time tau"""
        exp_op = self.fast_exp(tau)
        if callable(exp_op):  # Sparse case
            return exp_op(psi)
        else:
            return exp_op @ psi
    
    def intensity(self, psi):
        """Compute intensity for given state"""
        return np.real(np.vdot(psi, self.I_op @ psi))
    
    def apply_jump(self, psi, jump_index):
        """Apply jump operator to state"""
        return self.jump_ops[jump_index] @ psi

def excited_product_state_optimized(N):
    """
    Optimized creation of excited product state
    
    For large N, directly create the state vector without tensor products
    """
    if N <= 10:  # Use standard method for small systems
        from core import excited_product_state
        return excited_product_state(N)
    else:
        # For large systems, directly construct the state
        state_size = 2**N
        psi = np.zeros(state_size, dtype=complex)
        psi[-1] = 1.0  # |111...1⟩ state (all excited)
        return psi