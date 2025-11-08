using QuantumOptics, Plots

# -------------------------
# Parameters
# -------------------------
Γ = 0.3                # dephasing rate
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
# Calculate probability of being in initial state
# -------------------------
# Initial state density matrix
ρ_initial = dm(ψ0)
# Probability = Tr(ρ_initial * ρ_t) = real overlap
prob_initial = [real(tr(ρ_initial * ρ)) for ρ in ρt]

# -------------------------
# Plot
# -------------------------
p1 = plot(tout, ρ11, label="ρ11 (|↓⟩ population)", lw=2)
plot!(p1, tout, ρ22, label="ρ22 (|↑⟩ population)", lw=2)
plot!(p1, tout, coh, label="|ρ12| (coherence)", lw=2, linestyle=:dash)
xlabel!(p1, "Time")
ylabel!(p1, "Value")
title!(p1, "Pure Dephasing - Populations and Coherence")

# Plot probability of being in initial state
p2 = plot(tout, prob_initial, label="P(initial state)", lw=3, color=:red)
xlabel!(p2, "Time")
hline!(p2, [0.5], label="y = 0.5", lw=1, color=:black, linestyle=:dot)
ylabel!(p2, "Probability")
title!(p2, "Probability of Being in Initial State")

# Combine plots
plot(p1, p2, layout=(2,1), size=(600, 800))
