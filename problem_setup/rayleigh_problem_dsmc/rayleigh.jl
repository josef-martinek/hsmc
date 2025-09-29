###############################################################################
# Rayleigh Flow DSMC
# ---------------------------------------
# This script implements a minimal Direct Simulation Monte Carlo (DSMC) solver
# for the (impulsively started) Rayleigh flow problem: a quiescent rarefied gas
# confined between two infinite parallel plates where the lower plate suddenly
# begins to move tangentially with velocity Uw. Momentum then diffuses into the
# gas via molecular collisions. We compare / post-process macroscopic fields.
#
# Pedagogical Emphasis:
#  * Clear separation of DSMC steps: (1) Initialization, (2) Free molecular
#    flight (streaming / drift), (3) Boundary interactions, (4) Collisions
#    via acceptance–rejection in each spatial cell, (5) Sampling / averaging.
#  * Physical meaning of key non-dimensional numbers (Knudsen number Kn).
#  * Statistical weighting of simulated (sampling) particles (Ne).
#  * Construction of macroscopic fields (density, velocity, temperature).
#
# Notation Choices:
#  * All quantities are in nondimensional units with k_B = m = 1.
#  * Temperature T thus has units of (velocity)^2 and energy per particle.
#  * A Maxwellian with variance T is sampled using randn()*sqrt(T).
#
# Extensions you might explore after studying this code:
#  * Time averaging of macroscopic profiles after initial transient.
#  * Variance / confidence interval estimation across ensemble runs.
#  * Alternative collision pairing strategies (e.g., no time counter, NTC).
#  * Parallel domain decomposition for large particle counts.
#  * Energy / momentum conservation diagnostics.
###############################################################################

using Random                       # random numbers
using Statistics: mean             # basic statistics
using Base.Threads: @threads       # simple parallelism for collision loop

"""
    simulate(; Kn, Nsim, Uw, Tw, n0, N, Ncell, Nmft)

Run an ensemble of DSMC simulations of Rayleigh flow and return particle
positions, velocities, and a NamedTuple of parameters.

Arguments (keywords):
  Kn        :: Knudsen number = λ / L_z (sets domain height via L_z = λ / Kn)
  Nsim      :: number of statistically independent ensemble realizations
  Uw        :: tangential velocity of the lower wall (upper wall stationary & specular)
  Tw        :: (uniform) wall and initial gas temperature
  n0        :: initial (uniform) number density
  N         :: number of sampling (simulated) particles per realization
  Ncell     :: number of spatial bins (slabs) in z used for collisions / stats
  Nmft      :: physical simulation duration in units of mean-free times

Returns:
  (psm, vsm, params) where
    psm :: Array{Float64,3} (N,3,Nsim) final particle positions (x,y,z)
    vsm :: Array{Float64,3} (N,3,Nsim) final particle velocities (vx,vy,vz)
    params :: NamedTuple of derived parameters for postprocessing
"""
function simulate(;
    Kn=0.1,
    Nsim=3,
    Uw=0.2,       # lower wall (z=0) tangential velocity
    Tw=1.0,       # wall (and initial gas) temperature (k_B = m = 1 units)
    n0=0.001,     # initial number density
    N=100000,     # number of sampling particles (statistical, not real)
    Ncell=100,    # number of collision / sampling slabs in z
    Nmft=10,      # total simulated time in mean-free times
)
    # --- Derived kinetic / geometric parameters ---
    Nt = Nmft * 25                      # time steps (25 per mean-free time: heuristic)
    lambda_mfp = 1 / (sqrt(2) * π * n0) # mean free path for hard spheres (σ=π d^2, here σ absorbed)
    Lz = lambda_mfp / Kn                # choose domain height from desired Kn
    v_mean = (2 / sqrt(π)) * sqrt(2 * Tw) # mean speed of 3D Maxwellian with variance Tw per component
    tau = lambda_mfp / v_mean           # mean-free time (λ /  v )
    dt = Nmft * tau / Nt                # uniform time step
    dz = Lz / Ncell                     # slab thickness
    vol = Lz * dz * dz / Ncell          # effective cell volume (assuming square cross-section dz×dz)
    Ne = n0 * Lz * dz * dz / N          # number of physical molecules represented by one sampler

    #println("Knudsen number: $Kn")
    #println("Cells (Ncell) = $Ncell; expected sampling particles per cell ≈ $(round(N/Ncell, digits=1))")
    #println("Total simulated time: $Nmft mean-free times")
    #println("Weight per sampling particle Ne = $Ne (physical molecules)")

    # Allocate arrays to store final state of each realization
    psm = zeros(N, 3, Nsim)   # positions: columns 1,2,3 -> x,y,z
    vsm = zeros(N, 3, Nsim)   # velocities: columns 1,2,3 -> vx,vy,vz

    #base_seed = 17            # reproducible base seed; each sim uses an offset

    # --- Ensemble Loop -----------------------------------------------------
    # Each realization is statistically independent given different seed.
    for sim in 1:Nsim
        #println("Simulation $(sim) of $(Nsim)")
        #Random.seed!(base_seed + sim)

        # (1) Initialization: uniform spatial distribution & Maxwellian velocities
        x = dz .* rand(N)         # uniform in [0,dz] => periodic replication in x,y plane
        y = dz .* rand(N)
        z = Lz .* rand(N)         # uniform across channel height

        # Draw independent Gaussian components with std = sqrt(Tw)
        vx = randn(N) .* sqrt(Tw)
        vy = randn(N) .* sqrt(Tw)
        vz = randn(N) .* sqrt(Tw)

        # Upper bound guess for relative speed used in acceptance-rejection collisions.
        # Should exceed most sampled pair relative speeds to keep acceptance stable.
        v_rel_max = 6.0

        # Time integration loop (operator splitting: stream then collide)
        for it in 1:Nt
            # (2) Streaming: ballistic motion over dt (no forces assumed)
            @inbounds begin
                x .+= dt .* vx
                y .+= dt .* vy
                z .+= dt .* vz
            end

            # (3a) Upper wall (z = Lz): SPECULAR reflection: reverse normal component vz.
            hit_top = z .> Lz
            if any(hit_top)
                # Time after crossing top plane (positive if overshoot) to correct position.
                dt_ac = (z[hit_top] .- Lz) ./ vz[hit_top]
                vz[hit_top] .= -vz[hit_top]            # mirror reflection
                z[hit_top] .= Lz .+ dt_ac .* vz[hit_top] # rewind along new direction
            end

            # (3b) Lower wall (z = 0): THERMAL wall with imposed tangential drift Uw.
            # Particles striking the lower wall are re-emitted with a half-range Maxwellian
            # (only upward normal velocities) and tangential velocity biased by Uw.
            hit_bot = z .< 0
            if any(hit_bot)
                # Time AFTER collision (negative pre-collision time) to back-track to wall
                dt_ac = z[hit_bot] ./ vz[hit_bot]
                # Rewind to wall contact position BEFORE rethermalization
                x[hit_bot] .-= dt_ac .* vx[hit_bot]
                y[hit_bot] .-= dt_ac .* vy[hit_bot]
                Nbot = count(hit_bot)
                # Re-sample post-collision velocities:
                vx[hit_bot] .= sqrt(Tw) .* randn(Nbot)
                vy[hit_bot] .= sqrt(Tw) .* randn(Nbot) .+ Uw
                # Sample vz > 0 from half-range Maxwell distribution:
                # If U ~ Uniform(0,1): vz = sqrt(-2 T log U)
                vz[hit_bot] .= sqrt.(-2 * Tw .* log.(rand(Nbot)))
                # Advance forward remainder of step with new velocities
                x[hit_bot] .+= dt_ac .* vx[hit_bot]
                y[hit_bot] .+= dt_ac .* vy[hit_bot]
                z[hit_bot] .= dt_ac .* vz[hit_bot]
            end

            # (3c) Periodic boundaries in x,y: fold back into [0,dz). This effectively
            # simulates an infinite plate by tiling a small representative square.
            x .= mod.(x, dz)
            y .= mod.(y, dz)

            # (4) Collisions: Perform stochastic binary collisions inside each slab.
            # Hard-sphere model: Pick candidate pairs with acceptance probability
            #   P_accept = (|v_rel| / v_rel_max)
            # ensuring correct collision rate when M_cand chosen appropriately.
            N_collisions = 0
            @threads for j in 0:Ncell-1  # parallel over cells (race-safe: independent slabs)
                in_cell = (j * dz .< z) .& (z .< (j + 1) * dz)
                Nc = count(in_cell)
                Nc == 0 && continue
                idx = findall(in_cell)   # indices of particles in this slab

                # Expected number of collision attempts (Bird's DSMC scaling)
                M_cand = Int(ceil(Nc^2 * π * v_rel_max * Ne * dt / (2 * vol)))
                # Candidate collisions
                for _ in 1:M_cand
                    r_fac = rand()                            # uniform(0,1)
                    i_prop = idx[rand(1:length(idx))]         # random particle i
                    j_prop = idx[rand(1:length(idx))]         # random particle j
                    # Relative speed magnitude
                    dvx = vx[i_prop] - vx[j_prop]
                    dvy = vy[i_prop] - vy[j_prop]
                    dvz = vz[i_prop] - vz[j_prop]
                    v_rel = sqrt(dvx^2 + dvy^2 + dvz^2)
                    if v_rel > r_fac * v_rel_max
                        # (5) Process elastic hard-sphere collision: random post-collision
                        # direction in center-of-mass (COM) frame while preserving energy & momentum.
                        vx_cm = 0.5 * (vx[i_prop] + vx[j_prop])
                        vy_cm = 0.5 * (vy[i_prop] + vy[j_prop])
                        vz_cm = 0.5 * (vz[i_prop] + vz[j_prop])
                        cosθ = 2 * rand() - 1
                        sinθ = sqrt(1 - cosθ^2)
                        ϕ = 2 * π * rand()
                        vx_p = v_rel * sinθ * cos(ϕ)
                        vy_p = v_rel * sinθ * sin(ϕ)
                        vz_p = v_rel * cosθ
                        # Assign new velocities (equal masses): split relative contribution ±½
                        vx[i_prop] = vx_cm + 0.5 * vx_p; vx[j_prop] = vx_cm - 0.5 * vx_p
                        vy[i_prop] = vy_cm + 0.5 * vy_p; vy[j_prop] = vy_cm - 0.5 * vy_p
                        vz[i_prop] = vz_cm + 0.5 * vz_p; vz[j_prop] = vz_cm - 0.5 * vz_p
                        N_collisions += 1
                    end
                end
            end

            # (Optional) Another periodic wrap (safety) in x,y.
            x .= mod.(x, dz)
            y .= mod.(y, dz)
        end  # end time loop

        # Store final state for this realization (could extend to intermediate snapshots)
        psm[:, 1, sim] .= x; psm[:, 2, sim] .= y; psm[:, 3, sim] .= z
        vsm[:, 1, sim] .= vx; vsm[:, 2, sim] .= vy; vsm[:, 3, sim] .= vz
    end  # ensemble loop

    # Pack parameters for post-processing convenience
    params = (;
        Kn, Nsim, Uw, Tw, n0, N, Ncell, Nmft, Nt, lambda_mfp, Lz, v_mean, tau, dt, dz, vol, Ne
    )
    return psm, vsm, params
end

"""
        stats(ps, vs, params) -> NamedTuple

Compute macroscopic cell-averaged quantities (density, velocity components,
temperature) by binning particle data in the z direction.

Method:
    * Each particle contributes weight 1 to its cell count (later scaled by Ne / vol).
    * Velocity components are averaged arithmetically (sampling representation assumed uniform weight).
    * Temperature is computed from velocity variance (subtract mean squared, divide by 3 for 3 DOF).
Assumptions:
    * Monatomic gas (3 translational DOF, no internal energy modes).
    * Particle mass m = 1, Boltzmann constant k_B = 1.
"""
function stats(ps, vs, params)
    N, dim, Nsim = size(ps)
    @assert dim == 3
    (; Ncell, Lz, dz, Ne, vol) = params
    counts = zeros(Float64, Ncell)
    sum_vx = zeros(Float64, Ncell)
    sum_vy = zeros(Float64, Ncell)
    sum_vz = zeros(Float64, Ncell)
    sum_vx2 = zeros(Float64, Ncell)
    sum_vy2 = zeros(Float64, Ncell)
    sum_vz2 = zeros(Float64, Ncell)
    for sim in 1:Nsim
        z = view(ps, :, 3, sim)
        vx = view(vs, :, 1, sim)
        vy = view(vs, :, 2, sim)
        vz = view(vs, :, 3, sim)
        @inbounds for i in 1:N  # iterate particles once per simulation
            zval = z[i]
            if zval < 0 || zval > Lz
                continue
            end
            # Map z position to cell index (1-based) using floor division.
            cid = Int(clamp(fld(zval, dz) + 1, 1, Ncell))
            counts[cid] += 1
            vxi = vx[i]
            vyi = vy[i]
            vzi = vz[i]
            sum_vx[cid] += vxi
            sum_vy[cid] += vyi
            sum_vz[cid] += vzi
            sum_vx2[cid] += vxi^2
            sum_vy2[cid] += vyi^2
            sum_vz2[cid] += vzi^2
        end
    end
    # Ensemble average already implicit by summing all sims; no time-average yet.
    ρ = zeros(Float64, Ncell)
    ux = zeros(Float64, Ncell)
    uy = zeros(Float64, Ncell)
    uz = zeros(Float64, Ncell)
    T = zeros(Float64, Ncell)
    for c in 1:Ncell  # finalize averages per cell
        if counts[c] > 0
            inv = 1.0 / counts[c]
            ux[c] = sum_vx[c] * inv
            uy[c] = sum_vy[c] * inv
            uz[c] = sum_vz[c] * inv
            # number density
            ρ[c] = counts[c] * Ne / vol
            # temperature (monatomic, m=k_B=1)
            var_v =
                (sum_vx2[c] - sum_vx[c]^2 * inv) +
                (sum_vy2[c] - sum_vy[c]^2 * inv) +
                (sum_vz2[c] - sum_vz[c]^2 * inv)
            T[c] = (var_v * inv) / 3
        else
            ρ[c] = 0.0
            ux[c] = 0.0
            uy[c] = 0.0
            uz[c] = 0.0
            T[c] = 0.0
        end
    end
    zc = range(dz / 2, Lz - dz / 2; length=Ncell)
    return (z=collect(zc), density=ρ, ux=ux, uy=uy, uz=uz, T=T)
end

# Example usage
#using Plots
#ps, vs, params = simulate(; Kn=1.0, Nsim=10, Nmft=1)
#mac = stat_macros(ps, vs, params)
#plot(mac.z, mac.uy; xlabel="z", ylabel="u_y", label="u_y(z)")
