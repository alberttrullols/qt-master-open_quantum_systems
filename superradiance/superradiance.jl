using Random
using Statistics
using Plots

# Jump rate for collective decay: r(M) = Γ (J+M)(J−M+1)
rate(J, M, Γ) = Γ * (J + M) * (J - M + 1)

# One trajectory of pure Dicke superradiance
function trajectory(N; Γ=1.0, tmax=6.0, seed=0)
    Random.seed!(seed)
    J = N/2
    M = J
    t = 0.0

    times = [t]
    intensities = [rate(J, M, Γ)]

    while t < tmax && M > -J
        r = rate(J, M, Γ)
        dt = randexp() / r           # waiting time to next emission
        t += dt

        push!(times, t)
        M -= 1                       # jump M → M−1
        push!(intensities, rate(J, M, Γ))
    end

    return times, intensities
end

# Average over many trajectories
function monte_carlo(N; Γ=1.0, tmax=6.0, ntraj=200)
    all_t = Vector{Float64}[]
    all_I = Vector{Float64}[]

    for k in 1:ntraj
        t, I = trajectory(N; Γ=Γ, tmax=tmax, seed=k*100)
        push!(all_t, t)
        push!(all_I, I)
    end

    # Make a uniform time grid
    tgrid = range(0, tmax, length=500)
    Iavg = zeros(length(tgrid))

    for i in 1:length(tgrid)
        Ivals = Float64[]
        for (t, I) in zip(all_t, all_I)
            idx = searchsortedlast(t, tgrid[i])
            push!(Ivals, I[idx])
        end
        Iavg[i] = mean(Ivals)
    end

    return tgrid, Iavg
end

# ------------------------------
# Run and plot
# ------------------------------

N = 200
Γ = 1.0
t, I = monte_carlo(N; Γ=Γ, tmax=3, ntraj=10000)

plot(t, I,
     xlabel="Time",
     ylabel="Intensity I(t)",
     lw=2,
     title="Collective Superradiance (N=$N)",
     legend=false)
