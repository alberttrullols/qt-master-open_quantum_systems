import numpy as np
import scipy.linalg as la
from tqdm import tqdm
from core import setup_system, excited_product_state, interpolate_trajectory
from optimizations import (
    build_Gamma_optimized, 
    build_jump_ops_optimized,
    OptimizedEvolution,
    excited_product_state_optimized
)

def trajectory_original(N, positions, gamma0=1.0, k0=2*np.pi, tmax=4.0, seed=None):
    """
    Original trajectory simulation (for reference and comparison)
    """
    rng = np.random.default_rng(seed)
    
    # Set up system using core functions
    system = setup_system(N, positions, gamma0, k0)
    
    psi = excited_product_state(N)
    
    t = 0.0
    T = [0.0]
    I = [np.real(np.vdot(psi, system['I_op'] @ psi))]

    while t < tmax:
        # Compute rates
        rq = np.array([np.real(np.vdot(psi, (c.conj().T @ c) @ psi)) for c in system['jump_ops']])
        R = rq.sum()
        
        if R < 1e-14:
            break

        # Waiting time
        tau = -np.log(rng.random()) / R
        if t + tau > tmax:
            psi = la.expm(system['H_eff'] * (tmax - t)) @ psi
            psi /= la.norm(psi)
            t = tmax
            T.append(t)
            I.append(np.real(np.vdot(psi, system['I_op'] @ psi)))
            break

        # Evolve to jump
        psi = la.expm(system['H_eff'] * tau) @ psi
        psi /= la.norm(psi)
        t += tau

        # Choose and apply jump
        probs = np.maximum(rq/R, 0.0)
        probs = probs / probs.sum()
        q = rng.choice(len(system['jump_ops']), p=probs)
        psi = system['jump_ops'][q] @ psi
        psi /= la.norm(psi)

        T.append(t)
        I.append(np.real(np.vdot(psi, system['I_op'] @ psi)))

    return np.array(T), np.array(I)

def trajectory_optimized(N, positions, gamma0=1.0, k0=2*np.pi, tmax=4.0, seed=None):
    """
    Optimized trajectory simulation
    """
    rng = np.random.default_rng(seed)
    
    # Set up system with optimizations
    from core import lowering_ops, raising_ops, build_Iop
    
    lowers = lowering_ops(N)
    raises = raising_ops(N)
    Gamma = build_Gamma_optimized(positions, gamma0, k0)
    jump_ops = build_jump_ops_optimized(Gamma, lowers)
    I_op = build_Iop(Gamma, raises, lowers)
    
    # Create system dict for OptimizedEvolution
    CdagC = sum(c.conj().T @ c for c in jump_ops)
    H_eff = -0.5j * CdagC
    
    system_dict = {
        'H_eff': H_eff,
        'jump_ops': jump_ops,
        'I_op': I_op
    }
    
    # Create optimized evolution engine
    evolution = OptimizedEvolution(system_dict)
    
    # Initial state
    psi = excited_product_state_optimized(N)
    psi = psi / la.norm(psi)
    
    # Pre-allocate arrays
    max_steps = min(int(tmax * 1000), 5000)
    T = np.zeros(max_steps)
    I = np.zeros(max_steps)
    
    t = 0.0
    step = 0
    T[0] = 0.0
    I[0] = evolution.intensity(psi)
    step += 1

    # Evolution loop
    while t < tmax and step < max_steps:
        # Compute rates
        rates = evolution.compute_rates(psi)
        R = rates.sum()
        
        if R < 1e-14:
            break

        # Waiting time
        tau = -np.log(rng.random()) / R
        
        if t + tau > tmax:
            # Final evolution
            psi = evolution.evolve(psi, tmax - t)
            psi = psi / la.norm(psi)
            t = tmax
            T[step] = t
            I[step] = evolution.intensity(psi)
            step += 1
            break

        # Evolve to jump
        psi = evolution.evolve(psi, tau)
        psi = psi / la.norm(psi)
        t += tau

        # Choose and apply jump
        probs = np.maximum(rates / R, 0.0)
        probs = probs / probs.sum()
        q = rng.choice(len(jump_ops), p=probs)
        
        psi = evolution.apply_jump(psi, q)
        psi = psi / la.norm(psi)

        T[step] = t
        I[step] = evolution.intensity(psi)
        step += 1

    return T[:step], I[:step]

def ensemble(N, positions, gamma0=1.0, k0=2*np.pi, tmax=4.0, ntraj=200, engine='optimized', progress=True):
    """
    Run ensemble of trajectories
    """
    # Choose trajectory function
    if engine == 'original':
        trajectory_func = trajectory_original
    elif engine == 'optimized':
        trajectory_func = trajectory_optimized
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
    # Set up time grid
    tgrid = np.linspace(0, tmax, 400)
    I_avg = np.zeros_like(tgrid)
    
    desc = f"Trajectories ({engine})"
    iterator = tqdm(range(ntraj), desc=desc) if progress else range(ntraj)
    
    for seed in iterator:
        T, I = trajectory_func(N, positions, gamma0, k0, tmax, seed)
        I_traj = interpolate_trajectory(T, I, tgrid)
        I_avg += I_traj
    
    return tgrid, I_avg / ntraj