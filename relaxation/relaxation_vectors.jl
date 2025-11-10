using QuantumOptics
using Plots

b = SpinBasis(1//2)
σx = sigmax(b)
σy = sigmay(b)
σz = sigmaz(b)
σm = sigmam(b)    # lowering operator |0><1|

ψ0 = normalize(spinup(b) + spindown(b))
# ψ0 = normalize(spinup(b))
ρ0 = ψ0 ⊗ dagger(ψ0)

H = 1.0 * σz
Γ = 0.2                       # relaxation rate
c_ops = [sqrt(Γ) * σm]        # amplitude damping

tlist = 0:0.1:75
tout, ρt = timeevolution.master(tlist, ρ0, H, c_ops)


x = [real(expect(σx, ρ)) for ρ in ρt]
y = [real(expect(σy, ρ)) for ρ in ρt]
z = [real(expect(σz, ρ)) for ρ in ρt]


plot(tout, x, label="⟨σx⟩", lw=2)
plot!(tout, y, label="⟨σy⟩", lw=2)
plot!(tout, z, label="⟨σz⟩", lw=2)
xlabel!("Time")
ylabel!("Expectation values")
title!("Bloch vector evolution under relaxation (amplitude damping)")

savefig("bloch_relaxation_components.png")
