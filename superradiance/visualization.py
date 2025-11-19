import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def plot_static(t_data, I_data, labels=None, title="Superradiance", 
                filename=None, show=True):
    """
    Create static plot of superradiance data
    
    Parameters:
    -----------
    t_data : list or ndarray
        Time data (can be list of arrays for multiple datasets)
    I_data : list or ndarray  
        Intensity data (can be list of arrays for multiple datasets)
    labels : list, optional
        Labels for each dataset
    title : str
        Plot title
    filename : str, optional
        Save filename
    show : bool
        Whether to display plot
    """
    # Handle single dataset
    if not isinstance(t_data, list):
        t_data = [t_data]
        I_data = [I_data]
    
    # Default labels
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(t_data))]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
    for i, (t, I, label) in enumerate(zip(t_data, I_data, labels)):
        color = colors[i % len(colors)]
        plt.plot(t, I, color=color, linewidth=2.5, label=label)
    
    plt.xlim(0,max([np.max(t) for t in t_data])*1)
    plt.ylim(0, max([np.max(I) for I in I_data])*1.1)
    plt.xlabel("Time t (in units of 1/γ₀)", fontsize=12)
    plt.ylabel("Intensity I(t)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def animate_comparison(t_close, I_close, t_far, I_far, 
                      a_near, a_far, N, ntraj,
                      save_filename=None):
    """
    Create animated comparison of close vs far spacing
    """
    # Setup plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    max_I = max(np.max(I_close), np.max(I_far))
    ax.set_xlim(0, max(t_close[-1], t_far[-1]))
    ax.set_ylim(0, max_I * 1.1)
    ax.set_xlabel("Time t (in units of 1/γ₀)", fontsize=12)
    ax.set_ylabel("Intensity I(t)", fontsize=12)
    ax.set_title(f"Superradiance Effect: N={N} atoms, {ntraj} trajectories", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    line_close, = ax.plot([], [], 'b-', linewidth=2.5, label=f'Close spacing ({a_near} λ)')
    line_far, = ax.plot([], [], 'r-', linewidth=2.5, label=f'Far spacing ({a_far} λ)')
    ax.legend(fontsize=12)
    
    # Text for current values
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        total_frames = 200
        progress = frame / total_frames
        
        n_close = min(int(progress * len(t_close)), len(t_close))
        n_far = min(int(progress * len(t_far)), len(t_far))
        
        if n_close > 1:
            line_close.set_data(t_close[:n_close], I_close[:n_close])
        if n_far > 1:
            line_far.set_data(t_far[:n_far], I_far[:n_far])
        
        # Update text
        if n_close > 0 and n_far > 0:
            t_current = max(t_close[n_close-1] if n_close > 0 else 0,
                           t_far[n_far-1] if n_far > 0 else 0)
            I_close_current = I_close[n_close-1] if n_close > 0 else 0
            I_far_current = I_far[n_far-1] if n_far > 0 else 0
            
            text.set_text(f'Time: {t_current:.2f}\n'
                         f'I(close): {I_close_current:.4f}\n'
                         f'I(far): {I_far_current:.4f}')
        
        return line_close, line_far, text
    
    anim = FuncAnimation(fig, animate, frames=400, interval=50, 
                        blit=False, repeat=False)
    
    # Save animation if requested
    if save_filename:
        try:
            anim.save(save_filename, writer='ffmpeg', fps=20, dpi=150)
            print(f"Animation saved as {save_filename}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            # Fall back to static plot
            plot_static([t_close, t_far], [I_close, I_far],
                       labels=[f'Close ({a_near} λ)', f'Far ({a_far} λ)'],
                       title=f"Superradiance: N={N}, {ntraj} trajectories",
                       filename=save_filename.replace('.mp4', '_static.png'))
    
    return anim

def plot_comparison(t_close, I_close, t_far, I_far, 
                   a_near, a_far, N, ntraj,
                   filename=None, show=True):
    """
    Create static comparison plot
    
    Parameters similar to animate_comparison but for static plot
    """
    max_I_close = np.max(I_close)
    max_I_far = np.max(I_far)
    enhancement = max_I_close / max_I_far if max_I_far > 0 else float('inf')
    
    title = (f"Superradiance Effect: N={N} atoms, {ntraj} trajectories\n"
             f"Max intensity ratio: {enhancement:.2f}")
    
    plot_static([t_close, t_far], [I_close, I_far],
               labels=[f'Close spacing ({a_near} λ)', f'Far spacing ({a_far} λ)'],
               title=title, filename=filename, show=show)
    
    return enhancement

def quick_plot(t, I, title="Superradiance", filename=None):
    """Quick plotting function for single dataset"""
    plot_static(t, I, title=title, filename=filename)

def plot_scaling_analysis(N_values, times, speedups=None, filename=None):
    """
    Plot performance scaling analysis
    """
    if speedups is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Timing plot
    ax1.semilogy(N_values, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Atoms (N)')
    ax1.set_ylabel('Computation Time (seconds)')
    ax1.set_title('Performance Scaling')
    ax1.grid(True, alpha=0.3)
    
    # Theoretical scaling line
    if len(N_values) > 1:
        # Fit exponential: t = a * 2^(b*N)
        log_times = np.log2(times)
        coeffs = np.polyfit(N_values, log_times, 1)
        fit_times = 2**(coeffs[0] * np.array(N_values) + coeffs[1])
        ax1.plot(N_values, fit_times, 'r--', alpha=0.7, 
                label=f'2^({coeffs[0]:.1f}*N) scaling')
        ax1.legend()
    
    # Speedup plot
    if speedups is not None:
        ax2.plot(N_values, speedups, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Atoms (N)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Optimization Speedup')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
        ax2.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Scaling analysis saved as {filename}")
    
    plt.show()