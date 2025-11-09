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

# -------------------------
# Define three different initial states
# -------------------------
# 1. Excited state (spin-up)
ψ_excited = spinup(b)
ρ_excited = dm(ψ_excited)

# 2. Ground state (spin-down)
ψ_ground = spindown(b) 
ρ_ground = dm(ψ_ground)

# 3. Superposition with equal probability (in y-direction)
ψ_superposition = (spinup(b) + im*spindown(b))/sqrt(2)
ρ_superposition = dm(ψ_superposition)

# -------------------------
# Hamiltonian
# -------------------------
H = sigmaz(b)  # Pauli-Z Hamiltonian

# -------------------------
# Lindblad operator for pure dephasing
# -------------------------
L = sqrt(Γ) * sigmaz(b)

# -------------------------
# Solve master equation for each initial state
# -------------------------
# Excited state evolution
tout1, ρt_excited = timeevolution.master(tlist, ρ_excited, H, [L])

# Ground state evolution  
tout2, ρt_ground = timeevolution.master(tlist, ρ_ground, H, [L])

# Superposition state evolution
tout3, ρt_superposition = timeevolution.master(tlist, ρ_superposition, H, [L])

# -------------------------
# Calculate probability of remaining in initial state for each evolution
# -------------------------
# Initial state projectors
P_excited = dm(ψ_excited)
P_ground = dm(ψ_ground)  
P_superposition = dm(ψ_superposition)

# Probabilities = Tr(P_initial * ρ_t)
prob_initial_from_excited = [real(tr(P_excited * ρ)) for ρ in ρt_excited]
prob_initial_from_ground = [real(tr(P_ground * ρ)) for ρ in ρt_ground]
prob_initial_from_superposition = [real(tr(P_superposition * ρ)) for ρ in ρt_superposition]

# -------------------------
# Create the plot
# -------------------------
plot(tout1, prob_initial_from_excited, 
     label="Initial: |↑⟩ (Excited)", 
     lw=3, 
     color=:red,
     xlabel="Time",
     ylabel="Probability of Initial State",
     title="Probability of Remaining in Initial State under Pure Dephasing",
     size=(800, 600))

plot!(tout2, prob_initial_from_ground, 
      label="Initial: |↓⟩ (Ground)", 
      lw=3, 
      color=:blue)

plot!(tout3, prob_initial_from_superposition, 
      label="Initial: (|↑⟩ + |↓⟩)/√2 (Superposition)", 
      lw=3, 
      color=:green,
      linestyle=:dash)

# Add horizontal reference lines
hline!([0.0], label="P = 0", lw=1, color=:black, linestyle=:dot, alpha=0.5)
hline!([0.5], label="P = 0.5", lw=1, color=:black, linestyle=:dot, alpha=0.5)
hline!([1.0], label="P = 1", lw=1, color=:black, linestyle=:dot, alpha=0.5)

# Set y-axis limits for better visualization
ylims!((-0.05, 1.05))

# Save the plot
savefig("spin_up_probabilities_plot.png")
println("Plot saved as 'spin_up_probabilities_plot.png'")

# Debug: Let's check what's happening with the initial state probabilities
println("\nDebugging the initial state evolution:")
println("Probability of initial state for superposition:")
for i in [1, 50, 100, 500, 1000]
    println("t = $(tout3[i]): P(initial) = $(prob_initial_from_superposition[i])")
    println("  Off-diagonal: $(ρt_superposition[i].data[1,2])")
end
