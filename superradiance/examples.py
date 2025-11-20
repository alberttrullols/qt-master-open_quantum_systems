#!/usr/bin/env python3
# Import the main interface
from interface import simulate_superradiance, quick_simulation, compare_engines

# For more advanced usage
from engines import ensemble
from visualization import quick_plot
import numpy as np

def example_1_quick_start():
    """Example 1: Quick simulation with default parameters"""
    print("=" * 60)
    print("EXAMPLE 1: Quick Start")
    print("=" * 60)
    
    # Run a quick simulation
    results = quick_simulation(N=6, ntraj=50)
    
    print(f"Enhancement factor: {results['enhancement']:.2f}")
    print(f"Computation time: {results['computation_time']:.1f}s")

def example_2_custom_parameters():
    """Example 2: Custom parameters with animation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Parameters")
    print("=" * 60)
    
    # Custom simulation with specific parameters
    results = simulate_superradiance(
        N=10,                    # 10 atoms
        a_near=0.0075,         # Very close spacing
        a_far=4.5,             # Far spacing
        ntraj=200,             # 200 trajectories
        tmax=6.0,              # 6 time units
        engine='optimized',    # Use optimized engine
        animate=False,         # Animated plot
        save_files=True        # Save plot file
    )
    
    return results

def example_3_engine_comparison():
    """Example 3: Compare original vs optimized engines"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Engine Comparison")
    print("=" * 60)
    
    # Compare engines with small system for speed
    comparison = compare_engines(N=5, ntraj=20)
    
    print(f"\nSpeedup achieved: {comparison['speedup']:.2f}x")

def example_4_advanced_usage():
    """Example 4: Advanced usage with direct engine access"""
    print("\n" + "=" * 60) 
    print("EXAMPLE 4: Advanced Usage")
    print("=" * 60)
    
    # Set up custom positions
    N = 6
    positions = np.array([[i*0.02, 0, 0] for i in range(N)], dtype=float)
    
    # Run ensemble directly
    print("Running ensemble with custom positions...")
    t, I = ensemble(N, positions, ntraj=50, engine='optimized', progress=True)
    
    # Plot results
    quick_plot(t, I, title=f"Custom Positions: N={N}")
    
    print(f"Max intensity: {np.max(I):.4f}")
    print(f"Peak time: {t[np.argmax(I)]:.2f}")

def example_5_scaling_test():
    """Example 5: Test performance scaling"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Performance Scaling")  
    print("=" * 60)
    
    import time
    
    N_values = [4, 5, 6, 7]
    times = []
    
    for N in N_values:
        print(f"Testing N={N}...")
        
        start = time.time()
        results = quick_simulation(N=N, ntraj=20)
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"  N={N}: {elapsed:.2f}s, enhancement={results['enhancement']:.2f}")
    
    # Show scaling
    print(f"\nScaling from N=4 to N=7: {times[-1]/times[0]:.1f}x increase")

if __name__ == "__main__":
    print("Modular Superradiance Package Examples")
    print("=====================================")
    
    # Run examples
    try:
        results = example_2_custom_parameters()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()