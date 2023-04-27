"""
Compute and update multipliers `λ_z` for the augmented Lagrangian wit respect to `z=0` constraint.
"""
function ucmp_update_uc_lz_kernel(n::Int, T::Int, max_limit::Float64, z::CuDeviceArray{Float64,2}, lz::CuDeviceArray{Float64,2}, beta::Float64)
    tx = threadIdx().x
    bx = blockIdx().x
    if bx <= n && tx <= T
        @inbounds begin
            lz[bx,tx] = max(-max_limit, min(max_limit, lz[bx,tx] + (beta * z[bx,tx])))
        end
    end
    return
end

function admm_update_lz(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    admm_update_lz(env, mod.mpmodel, device)

    ngen = mod.mpmodel.models[1].grid_data.ngen
    par = env.params
    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution

    # update lz for uc binary solution
    nthreads = 2^ceil(Int, log2(3*mod.mpmodel.len_horizon))
    lztime = CUDA.@timed @cuda threads=nthreads blocks=ngen ucmp_update_uc_lz_kernel(ngen, 3*mod.mpmodel.len_horizon, par.MAX_MULTIPLIER, uc_sol.z_curr, uc_sol.lz, par.beta)

    # update lz for v ramp solution
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        lztime = CUDA.@timed @cuda threads=64 blocks=(div(ngen-1, 64)+1) update_lz_kernel(ngen, par.MAX_MULTIPLIER, vr_sol.z_curr, vr_sol.lz, par.beta)
    end
    return
end