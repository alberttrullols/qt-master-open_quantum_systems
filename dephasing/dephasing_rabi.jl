using QuantumOptics, Plots

Ω = 1.0               # Rabi frequency
Γ = 0.2               # dephasing rate
tlist = 0:0.01:20.0   # time array

b = SpinBasis(1//2)           # 2-level system

ψ0 = spindown(b)
ρ0 = dm(ψ0)

H = (Ω/2) * (sigmap(b) + sigmam(b))

L = sqrt(Γ) * sigmaz(b)

t, ρt = timeevolution.master(tlist, ρ0, H, [L])

ρ11 = [real(expect(dm(spindown(b)), ρ)) for ρ in ρt]
ρ22 = [real(expect(dm(spinup(b)), ρ)) for ρ in ρt]
coh = [abs(expect(sigmam(b), ρ)) for ρ in ρt]

p = plot(t, ρ22, label="P(excited)", lw=2, color=:red)
hline!(p, [0.5], label="y = 0.5", lw=1, color=:black, linestyle=:dash)
xlabel!(p, "Time")
ylabel!(p, "Probability of excited state")
title!(p, "Excited State Probability - Damped Rabi Oscillations (pure dephasing)")

# Display the plot
display(p)

# Optionally save the plot
savefig(p, "dephasing_plot.png")
