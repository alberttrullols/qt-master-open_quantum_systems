"""
Simple Interface for Superradiance Simulation
=============================================
This module provides easy-to-use functions for running superradiance simulations.
"""

import numpy as np
import time
from engines import ensemble
from visualization import plot_comparison, animate_comparison, quick_plot

def simulate_superradiance(N, a_near=0.01, a_far=1.0, ntraj=100, tmax=6.0, 
                          engine='optimized', animate=False, save_files=False):
    """
    Complete superradiance simulation with visualization
    
    Parameters:
    -----------
    N : int
        Number of atoms
    a_near : float
        Close spacing in units of wavelength
    a_far : float  
        Far spacing in units of wavelength
    ntraj : int
        Number of Monte Carlo trajectories
    tmax : float
        Maximum simulation time
    engine : str
        Simulation engine ('original' or 'optimized')
    animate : bool
        Create animation instead of static plot
    save_files : bool
        Save plot/animation files
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 't_close', 'I_close': close spacing results
        - 't_far', 'I_far': far spacing results  
        - 'enhancement': intensity enhancement factor
        - 'computation_time': total computation time
    """
    print(f"Superradiance Simulation")
    print(f"========================")
    print(f"N = {N} atoms")
    print(f"Close spacing: {a_near} λ")
    print(f"Far spacing: {a_far} λ") 
    print(f"Trajectories: {ntraj}")
    print(f"Engine: {engine}")
    print()
    
    # Setup positions
    positions_near = np.array([[i*a_near, 0, 0] for i in range(N)], dtype=float)
    positions_far = np.array([[i*a_far, 0, 0] for i in range(N)], dtype=float)
    
    # Run simulations
    start_time = time.time()
    
    print("Computing close spacing...")
    t_close, I_close = ensemble(N, positions_near, tmax=tmax, ntraj=ntraj, engine=engine)
    mid_time = time.time()
    
    print("Computing far spacing...")
    t_far, I_far = ensemble(N, positions_far, tmax=tmax, ntraj=ntraj, engine=engine)
    end_time = time.time()
    
    computation_time = end_time - start_time
    print(f"\nTotal computation time: {computation_time:.1f}s")
    
    # Calculate results
    max_I_close = np.max(I_close)
    max_I_far = np.max(I_far)
    enhancement = max_I_close / max_I_far if max_I_far > 0 else float('inf')
    
    print(f"Max intensity (close): {max_I_close:.4f}")
    print(f"Max intensity (far): {max_I_far:.4f}")
    print(f"Enhancement factor: {enhancement:.2f}")
    
    # Create visualization
    if animate:
        filename = f"superradiance_N{N}_ntraj{ntraj}.mp4" if save_files else None
        anim = animate_comparison(t_close, I_close, t_far, I_far, 
                                a_near, a_far, N, ntraj, filename)
    else:
        filename = f"superradiance_N{N}_ntraj{ntraj}.png" if save_files else None
        plot_comparison(t_close, I_close, t_far, I_far, 
                       a_near, a_far, N, ntraj, filename)
    
    # Return results
    results = {
        't_close': t_close,
        'I_close': I_close,
        't_far': t_far,
        'I_far': I_far,
        'enhancement': enhancement,
        'computation_time': computation_time,
        'parameters': {
            'N': N,
            'a_near': a_near,
            'a_far': a_far,
            'ntraj': ntraj,
            'tmax': tmax,
            'engine': engine
        }
    }
    
    return results

def quick_simulation(N, ntraj=50, engine='optimized'):
    """
    Quick simulation with default parameters
    
    Parameters:
    -----------
    N : int
        Number of atoms
    ntraj : int
        Number of trajectories
    engine : str
        Simulation engine
    
    Returns:
    --------
    results : dict
        Simulation results
    """
    return simulate_superradiance(N=N, ntraj=ntraj, engine=engine)

def compare_engines(N, ntraj=20, spacings=None):
    """
    Compare original vs optimized engines
    
    Parameters:
    -----------
    N : int
        Number of atoms
    ntraj : int
        Number of trajectories (should be small for fair comparison)
    spacings : tuple, optional
        (a_near, a_far) spacing values
    
    Returns:
    --------
    comparison : dict
        Comparison results including timing and accuracy
    """
    if spacings is None:
        a_near, a_far = 0.01, 1.0
    else:
        a_near, a_far = spacings
    
    print(f"Engine Comparison: N={N}, ntraj={ntraj}")
    print("=" * 50)
    
    # Run original engine
    print("Running original engine...")
    start = time.time()
    results_original = simulate_superradiance(N, a_near, a_far, ntraj, 
                                            engine='original', animate=False, 
                                            save_files=False)
    time_original = time.time() - start
    
    # Run optimized engine
    print("\nRunning optimized engine...")
    start = time.time()
    results_optimized = simulate_superradiance(N, a_near, a_far, ntraj,
                                             engine='optimized', animate=False,
                                             save_files=False)
    time_optimized = time.time() - start
    
    # Compare results
    speedup = time_original / time_optimized
    
    # Check accuracy (compare intensity curves)
    I_diff_close = np.max(np.abs(results_original['I_close'] - results_optimized['I_close']))
    I_diff_far = np.max(np.abs(results_original['I_far'] - results_optimized['I_far']))
    
    print(f"\nComparison Results:")
    print(f"Original time: {time_original:.2f}s")
    print(f"Optimized time: {time_optimized:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Max difference (close): {I_diff_close:.6f}")
    print(f"Max difference (far): {I_diff_far:.6f}")
    print(f"Accuracy maintained: {'✓' if max(I_diff_close, I_diff_far) < 1e-3 else '✗'}")
    
    return {
        'speedup': speedup,
        'time_original': time_original,
        'time_optimized': time_optimized,
        'accuracy_close': I_diff_close,
        'accuracy_far': I_diff_far,
        'results_original': results_original,
        'results_optimized': results_optimized
    }

def single_trajectory_demo(N, spacing=0.01, tmax=4.0, engine='optimized'):
    """
    Demonstrate single trajectory evolution
    
    Parameters:
    -----------
    N : int
        Number of atoms
    spacing : float
        Inter-atomic spacing
    tmax : float
        Maximum time
    engine : str
        Simulation engine
    
    Returns:
    --------
    t, I : ndarray
        Time and intensity arrays
    """
    from engines import trajectory_optimized, trajectory_original
    
    positions = np.array([[i*spacing, 0, 0] for i in range(N)], dtype=float)
    
    if engine == 'optimized':
        t, I = trajectory_optimized(N, positions, tmax=tmax, seed=42)
    else:
        t, I = trajectory_original(N, positions, tmax=tmax, seed=42)
    
    quick_plot(t, I, title=f"Single Trajectory: N={N}, spacing={spacing}λ")
    
    return t, I