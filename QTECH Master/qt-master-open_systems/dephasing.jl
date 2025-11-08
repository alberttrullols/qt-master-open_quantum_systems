using QuantumOptics, Plots

# -------------------------
# Parameters
# -------------------------
Γ = 0.5                # dephasing rate
tlist = 0:0.01:10.0    # time array

# -------------------------
# Define Hilbert space
# -------------------------
b = SpinBasis(1//2)     # two-level system

# Initial state: superposition
ψ0 = (spinup(b) + spindown(b))/sqrt(2)
ρ0 = dm(ψ0)         # initial density matrix

# -------------------------
# Hamiltonian (optional, here zero)
# -------------------------
H = sigmaz(b)               # Pauli-Z Hamiltonian

# -------------------------
# Lindblad operator for pure dephasing
# -------------------------
L = sqrt(Γ) * sigmaz(b)

# -------------------------
# Solve master equation
# -------------------------
tout, ρt = timeevolution.master(tlist, ρ0, H, [L])

# -------------------------
# Extract populations and coherence
# -------------------------
ρ11 = [real(expect(dm(spindown(b)), ρ)) for ρ in ρt]
ρ22 = [real(expect(dm(spinup(b)), ρ)) for ρ in ρt]
coh = [abs(ρ.data[1,2]) for ρ in ρt]

# -------------------------
# Plot
# -------------------------
plot(tout, ρ11, label="ρ11 (population)", lw=2)
plot!(tout, ρ22, label="ρ22 (population)", lw=2)
plot!(tout, coh, label="|ρ12| (coherence)", lw=2, linestyle=:dash)
xlabel!("Time")
ylabel!("Value")
title!("Pure Dephasing (Master Equation Solver)")
