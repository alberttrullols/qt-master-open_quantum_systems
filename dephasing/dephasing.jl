using QuantumOptics
using Plots

# ----------------------
# Define qubit and operators
# ----------------------
b = SpinBasis(1//2)
σx = sigmax(b)
σy = sigmay(b)
σz = sigmaz(b)

# ----------------------
# Initial state |+> = (|0> + |1>)/√2
# ----------------------
ψ0 = normalize(spinup(b) + spindown(b))
ρ0 = ψ0 ⊗ dagger(ψ0)

# ----------------------
# Hamiltonian and dephasing
# ----------------------
ω = 1.0         # rotation frequency
γ = 0.2         # dephasing rate
H = ω * σz
c_ops = [sqrt(γ) * σz]

# ----------------------
# Time evolution
# ----------------------
tlist = 0:0.1:10
tout, ρt = timeevolution.master(tlist, ρ0, H, c_ops)

# ----------------------
# Extract Bloch vector components
# ----------------------
x = [real(expect(σx, ρ)) for ρ in ρt]
y = [real(expect(σy, ρ)) for ρ in ρt]
z = [real(expect(σz, ρ)) for ρ in ρt]

# ----------------------
# Bloch sphere animation
# ----------------------
# Basic sphere
θ = 0:0.05:π
φ = 0:0.05:2π
X = [sin(th)*cos(ph) for th in θ, ph in φ]
Y = [sin(th)*sin(ph) for th in θ, ph in φ]
Z = [cos(th) for th in θ, ph in φ]

anim = @animate for i in 1:length(tout)
    plot(surface(X, Y, Z, alpha=0.1, color=:lightblue), legend=false)
    scatter!([0.0], [0.0], [0.0], color=:black, label=false) # origin
    # plot Bloch vector
    quiver!([0.0], [0.0], [0.0], quiver=([x[i]], [y[i]], [z[i]]),
            arrowsize=0.15, linewidth=2, color=:red)
    title!("Bloch vector spiraling under dephasing\nt = $(round(tout[i], digits=2))")
    xlims!(-1,1); ylims!(-1,1); zlims!(-1,1)
end

# Save as GIF
gif(anim, "bloch_spiral.gif", fps=20)