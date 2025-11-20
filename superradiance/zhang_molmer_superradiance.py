"""
Zhang–Mølmer style reduced jump simulator (dominant-mode approximation).

State: (N_atoms, J, M)
- start at fully inverted: J = N/2, M = +J
- collective jump channels from Gamma eigenmodes (default: largest only)
- optional independent/local decay channel

This code is designed to be compact, easy to translate to Julia, and fast for large N.
"""
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# -------------------------
# Utilities: Dicke algebra
# -------------------------
def A_JM(J, M):
    """Dicke matrix element A_{J,M} = sqrt( (J+M)*(J-M+1) )"""
    val = (J + M) * (J - M + 1)
    return np.sqrt(max(val, 0.0))

# -------------------------
# Geometry -> Gamma matrix
# -------------------------
def build_Gamma(positions, gamma0=1.0, k0=2*np.pi):
    N = len(positions)
    Gamma = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                Gamma[i, j] = gamma0
            else:
                r = np.linalg.norm(positions[i] - positions[j])
                kr = k0 * r
                # use sinc(kr) = sin(kr)/(kr); avoid division by zero
                if kr == 0:
                    Gamma[i, j] = gamma0
                else:
                    Gamma[i, j] = gamma0 * (np.sin(kr) / kr)
    # symmetrize for safety
    Gamma = 0.5 * (Gamma + Gamma.T)
    return Gamma

# -------------------------
# Diagonalize Gamma -> modes
# -------------------------
def gamma_modes(Gamma, n_modes=1):
    vals, vecs = la.eigh(Gamma)
    # eigenvalues in ascending order; take largest n_modes
    idx = np.argsort(vals)[::-1][:n_modes]
    lambdas = vals[idx]
    vecs_sel = vecs[:, idx]   # columns are eigenvectors
    # normalize eigenvectors so that ||v||=1 (la.eigh already gives orthonormal)
    return lambdas, vecs_sel

# -------------------------
# Effective collective amplitude for symmetric lowering
# -------------------------
def symmetric_overlap(v):
    """S_q = sum_j v_j (overlap of eigenvector with uniform vector)"""
    return np.sum(v)

# -------------------------
# Reduced-jump trajectory
# -------------------------
def run_trajectory_zm(N_atoms, lambdas, vecs, gamma_local=0.0, tmax=5.0, rng=None):
    """
    lambdas: array of eigenvalues [lambda_q]
    vecs: N_atoms x n_modes matrix of eigenvectors (columns)
    gamma_local: per-atom independent-decay rate
    returns times_list, intensity_list, jumps
    """
    if rng is None:
        rng = np.random.default_rng()

    # initial Dicke ladder: J = N/2, M = +J
    J = N_atoms / 2.0
    M = J
    N = N_atoms
    t = 0.0

    times = [0.0]
    # intensity from modes: sum_q lambda_q * |S_q|^2 * A_JM^2
    # compute S_q overlaps:
    S = np.array([symmetric_overlap(vecs[:, q]) for q in range(vecs.shape[1])])
    # note: if vecs are orthonormal, S may be small unless mode is symmetric.

    def intensity_current(Jc, Mc):
        # intensity contribution of each mode q given (J,M)
        Ajm = A_JM(Jc, Mc)
        if Ajm == 0:
            return 0.0
        Iq = lambdas * (np.abs(S)**2) * (Ajm**2)
        return np.real(np.sum(Iq))

    intensities = [intensity_current(J, M)]
    jump_record = []  # records (t, kind, mode_index)

    # For local decay approximate rate: gamma_local * (number excitations)
    # number excitations = J + M
    while t < tmax and (M > -J):
        # compute instantaneous rates:
        # collective-mode rates r_q(M) = lambda_q * |S_q|^2 * A_JM^2
        Ajm = A_JM(J, M)
        if Ajm == 0:
            break
            
        r_collective = lambdas * (np.abs(S)**2) * (Ajm**2)   # array per mode
        total_collective = float(np.sum(r_collective))

        # local decay rate:
        n_ex = J + M
        r_local = gamma_local * n_ex

        Rtot = total_collective + r_local
        if Rtot <= 1e-12:  # Use small threshold instead of zero
            # no jumps possible
            break

        # waiting time
        tau = rng.exponential(1.0 / Rtot)
        if t + tau > tmax:
            # advance & sample intensity at final time
            t = tmax
            times.append(t)
            intensities.append(intensity_current(J, M))
            break

        # advance
        t += tau

        # choose which channel
        r_all = np.concatenate((r_collective, np.array([r_local])))
        probs = r_all / Rtot
        
        # Handle numerical issues
        probs = np.maximum(probs, 0.0)
        probs = probs / np.sum(probs)
        
        choice = rng.choice(len(r_all), p=probs)

        if choice < len(lambdas):
            # collective mode jump q
            q = choice
            # apply collective jump: M -> M-1, J unchanged (dominant-mode approx)
            M = M - 1
            jump_record.append((t, 'collective', int(q)))
        else:
            # local decay jump: M->M-1
            M = M - 1
            # optionally, local decay can change J; here we keep J unchanged (simple approx)
            jump_record.append((t, 'local', None))

        times.append(t)
        intensities.append(intensity_current(J, M))

    return np.array(times), np.array(intensities), jump_record

# -------------------------
# Ensemble average
# -------------------------
def ensemble_zm(N_atoms, positions, gamma0=1.0, k0=2*np.pi, n_modes=1,
                gamma_local=0.0, tmax=5.0, ntraj=200, seed=0):
    Gamma = build_Gamma(positions, gamma0=gamma0, k0=k0)
    lambdas, vecs = gamma_modes(Gamma, n_modes)
    tgrid = np.linspace(0.0, tmax, 500)
    Iacc = np.zeros_like(tgrid)

    print(f"Running {ntraj} trajectories...")
    print(f"Largest eigenvalue: {lambdas[0]:.4f}")
    
    rng = np.random.default_rng(seed)
    for n in range(ntraj):
        if (n + 1) % 100 == 0:
            print(f"  Progress: {n+1}/{ntraj}")
            
        # for variation, use independent seeds per trajectory
        subrng = np.random.default_rng(rng.integers(1<<30))
        times, intensities, jumps = run_trajectory_zm(N_atoms, lambdas, vecs,
                                                     gamma_local=gamma_local, tmax=tmax, rng=subrng)
        
        # Check for valid trajectory data
        if len(times) > 1 and len(intensities) > 1:
            # piecewise-constant intensity between recorded times; map to tgrid
            idx = np.searchsorted(times, tgrid, side='right') - 1
            idx = np.clip(idx, 0, len(intensities) - 1)  # Safe bounds
            Iinterp = intensities[idx]
            Iacc += Iinterp

    Imean = Iacc / ntraj
    return tgrid, Imean, Gamma, lambdas, vecs

# -------------------------
# Example usage & plotting
# -------------------------
if __name__ == "__main__":
    # Parameters
    N = 10
    a = 0.0075    # spacing in units of lambda; small -> collective
    a_far = 4.5 # spacing larger -> weaker collectivity
    gamma0 = .01
    k0 = 2 * np.pi  # lambda = 1
    tmax = 6.0
    ntraj = 200

    print(f"Zhang-Mølmer Superradiance Simulation")
    print(f"N_atoms = {N}, tmax = {tmax}, ntraj = {ntraj}")
    print("="*50)

    # Build positions (1D chain)
    pos_near = np.array([[i * a, 0.0, 0.0] for i in range(N)])
    pos_far  = np.array([[i * a_far, 0.0, 0.0] for i in range(N)])

    print("Running near case (a = 0.05λ)...")
    # Near case
    t_near, I_near, Gamma_near, lambdas_near, vecs_near = ensemble_zm(
        N, pos_near, gamma0=gamma0, k0=k0, n_modes=1, gamma_local=0.0, tmax=tmax, ntraj=ntraj, seed=1
    )

    print("\nRunning far case (a = 0.6λ)...")
    # Far case
    t_far, I_far, Gamma_far, lambdas_far, vecs_far = ensemble_zm(
        N, pos_far, gamma0=gamma0, k0=k0, n_modes=1, gamma_local=0.0, tmax=tmax, ntraj=ntraj, seed=2
    )

    # Create and save plot
    print("\nGenerating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(t_near, I_near, 'b-', linewidth=2, label=f"Near: a = {a}λ")
    plt.plot(t_far,  I_far,  'r-', linewidth=2, label=f"Far: a = {a_far}λ")
    plt.xlabel("Time (γ⁻¹)", fontsize=12)
    plt.ylabel("Intensity (arbitrary units)", fontsize=12)
    plt.title(f"Zhang–Mølmer Superradiance (N = {N} atoms)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, tmax)
    plt.ylim(bottom=0)
    plt.tight_layout()
    
    # Save plot instead of showing
    output_file = 'zhang_molmer_superradiance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")

    print(f"\nResults:")
    print(f"Largest eigenvalue (near): {lambdas_near[0]:.4f}")
    print(f"Largest eigenvalue (far):  {lambdas_far[0]:.4f}")
    print(f'Largest eigenvalue happens at time: {t_near[np.argmax(I_near)]:.4f} (near), {t_far[np.argmax(I_far)]:.4f} (far)')
    print(f"Enhancement factor: {lambdas_near[0]/lambdas_far[0]:.2f}")
    
    # Additional analysis
    print(f"\nPeak intensity (near): {np.max(I_near):.4f}")
    print(f"Peak intensity (far):  {np.max(I_far):.4f}")
    print(f"Intensity enhancement: {np.max(I_near)/np.max(I_far):.2f}")
    
    plt.close()  # Clean up