"""
Shared Core Functions for Superradiance Simulation
==================================================
This module contains the common functions used across all versions
of the superradiance simulation.
"""

import numpy as np
import scipy.linalg as la

# Basic quantum operators for 2-level atoms
sm = np.array([[0, 1], [0, 0]], dtype=complex)    # lowering: |e⟩ → |g⟩
sp = np.array([[0, 0], [1, 0]], dtype=complex)    # raising: |g⟩ → |e⟩
id2 = np.eye(2, dtype=complex)                     # identity

def kronN(ops):
    """Compute tensor product of multiple operators"""
    out = np.array([[1]], dtype=complex)
    for A in ops:
        out = np.kron(out, A)
    return out

def single_site(op, i, N):
    """Create single-site operator in N-particle Hilbert space"""
    ops = [id2] * N
    ops[i] = op
    return kronN(ops)

def lowering_ops(N):
    """Create list of lowering operators for N atoms"""
    return [single_site(sm, i, N) for i in range(N)]

def raising_ops(N):
    """Create list of raising operators for N atoms"""  
    return [single_site(sp, i, N) for i in range(N)]

def build_Gamma(positions, gamma0=1.0, k0=2*np.pi):
    """
    Build distance-dependent decay matrix Γ_ij
    
    Parameters:
    -----------
    positions : array_like
        Atomic positions (N x 3 array)
    gamma0 : float
        Single-atom decay rate
    k0 : float
        Wave number (2π/λ)
    
    Returns:
    --------
    Gamma : ndarray
        Symmetric decay matrix
    """
    N = len(positions)
    Gamma = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                Gamma[i, i] = gamma0
            else:
                r = np.linalg.norm(positions[i] - positions[j])
                kr = k0 * r
                Gamma[i, j] = gamma0 * np.sinc(kr/np.pi)  # sin(kr)/(kr)
    
    return 0.5 * (Gamma + Gamma.T)  # Ensure symmetry

def build_jump_ops(Gamma, lower_ops):
    """
    Build collective jump operators from decay matrix
    
    Parameters:
    -----------
    Gamma : ndarray
        Decay matrix
    lower_ops : list
        List of single-atom lowering operators
    
    Returns:
    --------
    jump_ops : list
        List of collective jump operators c_q
    """
    vals, vecs = la.eigh(Gamma)
    ops = []
    
    for eigval, eigvec in zip(vals, vecs.T):
        if eigval > 1e-12:  # Only keep positive eigenvalues
            cq = np.zeros_like(lower_ops[0])
            for j, vj in enumerate(eigvec):
                cq += vj * lower_ops[j]
            ops.append(np.sqrt(eigval) * cq)
    
    return ops

def build_Iop(Gamma, raising_ops, lowering_ops):
    """
    Build intensity operator I = Σ_ij Γ_ij σ_i^+ σ_j^-
    
    Parameters:
    -----------
    Gamma : ndarray
        Decay matrix
    raising_ops : list
        List of raising operators
    lowering_ops : list
        List of lowering operators
    
    Returns:
    --------
    I_op : ndarray
        Intensity operator
    """
    N = Gamma.shape[0]
    I = np.zeros_like(lowering_ops[0])
    
    for i in range(N):
        for j in range(N):
            I += Gamma[i, j] * (raising_ops[i] @ lowering_ops[j])
    
    return I

def excited_product_state(N):
    """
    Create fully excited product state |e⟩^⊗N
    
    Parameters:
    -----------
    N : int
        Number of atoms
    
    Returns:
    --------
    psi : ndarray
        Normalized excited state vector
    """
    e = np.array([0, 1], dtype=complex)
    psi = e
    for _ in range(N-1):
        psi = np.kron(psi, e)
    return psi / la.norm(psi)

def setup_system(N, positions, gamma0=1.0, k0=2*np.pi):
    """
    Set up complete quantum system operators
    
    Parameters:
    -----------
    N : int
        Number of atoms
    positions : array_like
        Atomic positions
    gamma0 : float
        Single-atom decay rate
    k0 : float
        Wave number
    
    Returns:
    --------
    system : dict
        Dictionary containing all system operators:
        - 'lowers': lowering operators
        - 'raises': raising operators  
        - 'Gamma': decay matrix
        - 'jump_ops': collective jump operators
        - 'I_op': intensity operator
        - 'H_eff': effective Hamiltonian
    """
    # Build operators
    lowers = lowering_ops(N)
    raises = raising_ops(N)
    Gamma = build_Gamma(positions, gamma0, k0)
    jump_ops = build_jump_ops(Gamma, lowers)
    I_op = build_Iop(Gamma, raises, lowers)
    
    # Effective Hamiltonian
    CdagC = sum(c.conj().T @ c for c in jump_ops)
    H_eff = -0.5j * CdagC
    
    return {
        'lowers': lowers,
        'raises': raises, 
        'Gamma': Gamma,
        'jump_ops': jump_ops,
        'I_op': I_op,
        'H_eff': H_eff,
        'N': N,
        'positions': positions
    }

def interpolate_trajectory(T, I, tgrid):
    """
    Interpolate trajectory data onto regular time grid
    
    Parameters:
    -----------
    T : ndarray
        Trajectory time points
    I : ndarray
        Trajectory intensity values
    tgrid : ndarray
        Regular time grid
    
    Returns:
    --------
    I_interp : ndarray
        Interpolated intensity values
    """
    if len(T) <= 1:
        return np.zeros_like(tgrid)
    
    idx = np.searchsorted(T, tgrid, side="right") - 1
    idx = np.clip(idx, 0, len(I) - 1)
    return I[idx]