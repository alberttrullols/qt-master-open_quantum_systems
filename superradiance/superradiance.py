
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
"""
Superradiance Simulation via Monte Carlo Wave-Function Method
-------------------------------------------------------------
This script simulates collective spontaneous emission (superradiance) in a system of N two-level atoms using the quantum jump (Monte Carlo) approach.

- Each atom is a two-level system (|g⟩, |e⟩).
- Atoms interact via distance-dependent decay rates Γ_ij, modeling collective emission.
- The system evolves stochastically via quantum jumps and non-Hermitian effective Hamiltonian evolution.

1. Distance-dependent decay matrix:
   Γ_ij = γ₀ * sinc(k₀ * |r_i - r_j| / π)
   where γ₀ is the single-atom decay rate, k₀ = 2π/λ, and r_i are atom positions.

2. Jump operators (collective emission channels):
   c_q = Σ_j v_j^(q) σ_j^-
   where v_j^(q) are eigenvectors of Γ_ij, σ_j^- is the lowering operator.

3. Effective non-Hermitian Hamiltonian:
   H_eff = -i/2 Σ_q c_q† c_q

4. Intensity operator:
   I = Σ_ij Γ_ij σ_i^+ σ_j^-

Simulation Method:
- Start in the fully excited state |e⟩⊗N.
- Evolve under H_eff until a quantum jump occurs (random waiting time).
- Apply a jump operator c_q to the state.
- Repeat until final time t_max.
- Average intensity over many independent trajectories (Monte Carlo).
"""

# Operators for a single 2-level atom
# In the basis |g>, |e> = |0>, |1>
sm = np.array([[0, 1], [0, 0]], dtype=complex)    # lowering: |e> -> |g>
sp = np.array([[0, 0], [1, 0]], dtype=complex)    # raising: |g> -> |e>
id2 = np.eye(2, dtype=complex)

def kronN(ops):
    out = np.array([[1]], dtype=complex)
    for A in ops:
        out = np.kron(out, A)
    return out

def single_site(op, i, N):
    ops = [id2] * N
    ops[i] = op
    return kronN(ops)

def lowering_ops(N):
    return [single_site(sm, i, N) for i in range(N)]

def raising_ops(N):
    return [single_site(sp, i, N) for i in range(N)]

# Distance-dependent decay Γ_ij  (scalar free-space model)-
def build_Gamma(positions, gamma0=1.0, k0=2*np.pi):
    N = len(positions)
    Gamma = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Gamma[i,i] = gamma0
            else:
                r = np.linalg.norm(positions[i] - positions[j])
                kr = k0 * r
                Gamma[i,j] = gamma0 * np.sinc(kr/np.pi)  # sin(kr)/(kr)
    return 0.5*(Gamma + Gamma.T)

# Build jump operators c_q from Γ_ij
def build_jump_ops(Gamma, lower_ops):
    vals, vecs = la.eigh(Gamma)
    ops = []
    for eigval, eigvec in zip(vals, vecs.T):
        if eigval > 1e-12:
            cq = np.zeros_like(lower_ops[0])
            for j, vj in enumerate(eigvec):
                cq += vj * lower_ops[j]
            ops.append(np.sqrt(eigval) * cq)
    return ops

# Intensity operator I = Σ_ij Γ_ij σ_i^+ σ_j^-
def build_Iop(Gamma, raising_ops, lowering_ops):
    N = Gamma.shape[0]
    I = np.zeros_like(lowering_ops[0])
    for i in range(N):
        for j in range(N):
            I += Gamma[i,j] * (raising_ops[i] @ lowering_ops[j])
    return I

def excited_product_state(N):
    e = np.array([0,1], dtype=complex)
    psi = e
    for _ in range(N-1):
        psi = np.kron(psi, e)
    return psi / la.norm(psi)

def trajectory(N, positions, gamma0=1.0, k0=2*np.pi, tmax=4.0, seed=None):
    rng = np.random.default_rng(seed)

    lowers = lowering_ops(N)
    raises = raising_ops(N)

    Gamma = build_Gamma(positions, gamma0=gamma0, k0=k0)
    cs = build_jump_ops(Gamma, lowers)
    Iop = build_Iop(Gamma, raises, lowers)

    # effective H (no coherent part here; easy to add later)
    CdagC = sum(c.conj().T @ c for c in cs)
    Heff = -0.5j * CdagC

    psi = excited_product_state(N)

    t = 0.0
    T = [0.0]
    I = [np.real(np.vdot(psi, Iop @ psi))]

    while t < tmax:
        # rates
        rq = np.array([np.real(np.vdot(psi, (c.conj().T @ c) @ psi)) for c in cs])
        R = rq.sum()
        if R < 1e-14:
            break

        # waiting time
        tau = -np.log(rng.random()) / R
        if t + tau > tmax:
            psi = la.expm(Heff*(tmax - t)) @ psi
            psi /= la.norm(psi)
            t = tmax
            T.append(t)
            I.append(np.real(np.vdot(psi, Iop @ psi)))
            break

        # evolve to jump
        psi = la.expm(Heff * tau) @ psi
        psi /= la.norm(psi)
        t += tau

        # choose jump (ensure probabilities are non-negative)
        probs = np.maximum(rq/R, 0.0)  # clip negative values
        probs = probs / probs.sum()    # renormalize
        q = rng.choice(len(cs), p=probs)
        psi = cs[q] @ psi
        psi /= la.norm(psi)

        T.append(t)
        I.append(np.real(np.vdot(psi, Iop @ psi)))

    return np.array(T), np.array(I)

# defined outside ensemble to allow multiprocessing
def run_traj(args):
    N, positions, gamma0, k0, tmax, seed, tgrid = args
    T, I = trajectory(N, positions, gamma0, k0, tmax, seed)
    idx = np.searchsorted(T, tgrid, side="right") - 1
    idx[idx < 0] = 0
    return I[idx]

def ensemble(N, positions, gamma0=1.0, k0=2*np.pi, tmax=4.0, ntraj=200):
    import concurrent.futures
    from tqdm import tqdm
    max_workers = 1
    tgrid = np.linspace(0, tmax, 400)
    Iavg = np.zeros_like(tgrid)

    args_list = [(N, positions, gamma0, k0, tmax, seed, tgrid) for seed in range(ntraj)]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for I_traj in tqdm(executor.map(run_traj, args_list), total=ntraj, desc="Trajectories"):
            Iavg += I_traj
            results.append(I_traj)

    return tgrid, Iavg / ntraj

def plot_superradiance(N, a_near, a_far, ntraj=100):
    # Setup positions and calculate
    positions_near = np.array([[i*a_near, 0, 0] for i in range(N)], dtype=float)
    positions_far = np.array([[i*a_far, 0, 0] for i in range(N)], dtype=float)
    
    t_close, I_close = ensemble(N, positions_near, tmax=6.0, ntraj=ntraj)
    t_far, I_far = ensemble(N, positions_far, tmax=6.0, ntraj=ntraj)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_close, I_close, 'b-', label=f'Close ({a_near} λ)')
    plt.plot(t_far, I_far, 'r-', label=f'Far ({a_far} λ)')
    plt.xlabel('Time (1/γ₀)')
    plt.ylabel('Intensity')
    plt.title(f'Superradiance: N={N}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return t_close, I_close, t_far, I_far

# ----------------------------------------------------------
# Animation function for progressive plotting
# ----------------------------------------------------------
def animate_superradiance(N, a_near, a_far, ntraj=100):
    from matplotlib.animation import FuncAnimation
    import time
    
    # Setup positions
    positions_near = np.array([[i*a_near, 0, 0] for i in range(N)], dtype=float)
    positions_far = np.array([[i*a_far, 0, 0] for i in range(N)], dtype=float)
    
    # Calculate data with progress indication
    print(f"Calculating superradiance for N={N} atoms...")
    print(f"Close spacing: {a_near} λ, Far spacing: {a_far} λ")
    
    start_time = time.time()
    t_close, I_close = ensemble(N, positions_near, gamma0=1.0, k0=2*np.pi, tmax=6.0, ntraj=ntraj)
    mid_time = time.time()
    print(f"Close spacing calculation took {mid_time - start_time:.1f}s")

    t_far, I_far = ensemble(N, positions_far, gamma0=1.0, k0=2*np.pi, tmax=6.0, ntraj=ntraj)
    end_time = time.time()
    print(f"Far spacing calculation took {end_time - mid_time:.1f}s")
    
    # Print results
    max_I_close = np.max(I_close)
    max_I_far = np.max(I_far)
    print(f"\nResults:")
    print(f"Max intensity (close): {max_I_close:.4f}")
    print(f"Max intensity (far): {max_I_far:.4f}")
    if max_I_far > 0:
        print(f"Enhancement factor: {max_I_close/max_I_far:.2f}")
    
    # Setup animated plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, float(max(max_I_close, max_I_far) * 1.1))
    ax.set_xlabel("Time t (in units of 1/γ₀)", fontsize=12)
    ax.set_ylabel("Intensity I(t)", fontsize=12)
    ax.set_title(f"Superradiance Effect: N={N} atoms", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    line_close, = ax.plot([], [], 'b-', linewidth=2.5, label=f'Close spacing ({a_near} λ)')
    line_far, = ax.plot([], [], 'r-', linewidth=2.5, label=f'Far spacing ({a_far} λ)')
    ax.legend(fontsize=12)
    
    # Add text for real-time values
    text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        # Progressive revelation
        total_frames = 150
        progress = frame / total_frames
        
        n_close = min(int(progress * len(t_close)), len(t_close))
        n_far = min(int(progress * len(t_far)), len(t_far))
        
        if n_close > 1:
            line_close.set_data(t_close[:n_close], I_close[:n_close])
        if n_far > 1:
            line_far.set_data(t_far[:n_far], I_far[:n_far])
            
        # Update text with current values
        if n_close > 0 and n_far > 0:
            t_current = max(t_close[n_close-1] if n_close > 0 else 0, 
                           t_far[n_far-1] if n_far > 0 else 0)
            I_close_current = I_close[n_close-1] if n_close > 0 else 0
            I_far_current = I_far[n_far-1] if n_far > 0 else 0
            
            text.set_text(f'Time: {t_current:.2f}\nI(close): {I_close_current:.4f}\nI(far): {I_far_current:.4f}')
        
        return line_close, line_far, text
    
    anim = FuncAnimation(fig, animate, frames=300, interval=80, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

    # Save animation with parameters in filename
    filename = f"superradiance_N{N}_anear{a_near}_afar{a_far}_ntraj{ntraj}.mp4"
    try:
        anim.save(filename, writer='ffmpeg', fps=12)
        print(f"Animation saved as {filename}")
    except Exception as e:
        print(f"Could not save animation: {e}")

    return t_close, I_close, t_far, I_far

if __name__ == "__main__":
    # Run full animation
    N = 8
    a_near = 0.0075   # Very close -> strong superradiance
    a_far = 4.5     # Far apart -> independent emission
    ntraj =150

    animate_superradiance(N, a_near, a_far, ntraj=ntraj)
