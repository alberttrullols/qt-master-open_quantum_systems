using QuantumOptics
using Plots

# ----------------------
# Define qubit and operators
# ----------------------
b = SpinBasis(1//2)          # spin-1/2
σx = sigmax(b)
σy = sigmay(b)
σz = sigmaz(b)

# ----------------------
# Define initial state
# ----------------------
ψ0 = spinup(b) + spindown(b)
# ψ0 = spindown(b)
ψ0 = normalize(ψ0)
ρ0 = ψ0 ⊗ dagger(ψ0) 

# ----------------------
# Hamiltonian and collapse operator (dephasing)
# ----------------------
H = 1.0 * σz
γ = 0.2                        # dephasing rate
c_ops = [sqrt(γ) * σz]         # pure dephasing along z

# ----------------------
# Time evolution
# ----------------------
tlist = 0:0.1:10
tout, ρt = timeevolution.master(tlist, ρ0, H, c_ops)

# ----------------------
# Extract Bloch components
# ----------------------
x = [real(expect(σx, ρ)) for ρ in ρt]
y = [real(expect(σy, ρ)) for ρ in ρt]
z = [real(expect(σz, ρ)) for ρ in ρt]

# ----------------------
# Plot Bloch vector components
# ----------------------
plot(tout, x, label="⟨σx⟩", lw=2)
plot!(tout, y, label="⟨σy⟩", lw=2)
plot!(tout, z, label="⟨σz⟩", lw=2)
xlabel!("Time")
ylabel!("Expectation values")
title!("Bloch vector evolution under dephasing")