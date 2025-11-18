using Random
using Statistics
using Plots

# Generate independent decay times: each atom decays once with rate γ
function independent_decay_times(N; γ=1.0, seed=0)
    Random.seed!(seed)
    return randexp(N) ./ γ
end

# Manual histogram → intensity
function intensity_from_times(times; tmax=6.0, nbins=200)
    Δt = tmax / nbins
    bins = collect(0:Δt:(tmax-Δt))             # left edges
    counts = zeros(nbins)

    # Fill bins manually
    for τ in times
        if τ < tmax
            idx = Int(floor(τ / Δt)) + 1
            counts[idx] += 1
        end
    end

    intensity = counts ./ Δt                   # counts → rate
    bin_centers = bins .+ Δt/2

    return bin_centers, intensity
end

# Monte Carlo averaging over many independent-emission runs
function monte_carlo_intensity(N; γ=1.0, tmax=6.0, ntraj=200)
    tgrid = nothing
    intensities = []

    for k in 1:ntraj
        times = independent_decay_times(N; γ=γ, seed=100*k)
        t, I = intensity_from_times(times; tmax=tmax)
        if tgrid === nothing
            tgrid = t
        end
        push!(intensities, I)
    end

    Iavg = mean(reduce(hcat, intensities), dims=2)[:]
    return tgrid, Iavg
end

# --------------------------
# Run and plot
# --------------------------

N = 200
γ = 1.0

t, I = monte_carlo_intensity(N; γ=γ, tmax=3, ntraj=10000)

plot(
    t, I,
    lw=2,
    xlabel="Time",
    ylabel="Intensity I(t)",
    title="Independent Irradiance (N=$N)",
    legend=false
)
