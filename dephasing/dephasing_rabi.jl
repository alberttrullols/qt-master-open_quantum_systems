using QuantumOptics, Plots

# -------------------------
# Parameters
# -------------------------
Ω = 1.0               # Rabi frequency
Γ = 0.2               # dephasing rate
tlist = 0:0.01:20.0   # time array

# -------------------------
# Hilbert space
# -------------------------
b = SpinBasis(1//2)           # 2-level system

# Initial state: ground state
ψ0 = spindown(b)
ρ0 = dm(ψ0)

# -------------------------
# Hamiltonian for Rabi drive
# H = (Ω/2) * (σ_+ + σ_-)
# -------------------------
H = (Ω/2) * (sigmap(b) + sigmam(b))

# -------------------------
# Lindblad operator for dephasing
# -------------------------
L = sqrt(Γ) * sigmaz(b)

# -------------------------
# Solve master equation
# -------------------------
t, ρt = timeevolution.master(tlist, ρ0, H, [L])

# -------------------------
# Extract populations and coherence
# -------------------------
ρ11 = [real(expect(dm(spindown(b)), ρ)) for ρ in ρt]
ρ22 = [real(expect(dm(spinup(b)), ρ)) for ρ in ρt]
coh = [abs(expect(sigmam(b), ρ)) for ρ in ρt]

# -------------------------
# Plot results
# -------------------------
p = plot(t, ρ22, label="P(excited)", lw=2, color=:red)
hline!(p, [0.5], label="y = 0.5", lw=1, color=:black, linestyle=:dash)
xlabel!(p, "Time")
ylabel!(p, "Probability of excited state")
title!(p, "Excited State Probability - Damped Rabi Oscillations (pure dephasing)")

# Display the plot
display(p)

# Optionally save the plot
savefig(p, "dephasing_plot.png")
