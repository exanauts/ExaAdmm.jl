"""
Update multipliers λ for consensus constraints, `x - xbar + z = 0`.
"""
function ucmp_update_uc_l_kernel(
    n::Int, T::Int, 
    l::CuDeviceArray{Float64,2}, z::CuDeviceArray{Float64,2}, lz::CuDeviceArray{Float64,2},
    beta::Float64
)
    tx = threadIdx().x
    bx = blockIdx().x
    if bx <= n && tx <= T
        @inbounds begin
            l[bx,tx] = -(lz[bx,tx] + beta*z[bx,tx])
        end
    end
    return
end

function admm_update_l(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    admm_update_l(env, mod.mpmodel, device)

    # update l for uc_solution
    ngen = mod.mpmodel.models[1].grid_data.ngen
    par = env.params
    uc_sol = mod.uc_solution
    uc_sol.l_curr .= -(uc_sol.lz .+ par.beta .* uc_sol.z_curr)
    nthreads = 2^ceil(Int, log2(3*mod.mpmodel.len_horizon))
    ltime = CUDA.@timed @cuda threads=nthreads blocks=ngen ucmp_update_uc_l_kernel(ngen, 3*mod.mpmodel.len_horizon, uc_sol.l_curr, uc_sol.z_curr, uc_sol.lz, par.beta)

    # update l for v ramp solution
    vr_sols = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        ltime = CUDA.@timed @cuda threads=64 blocks=(div(ngen-1, 64)+1) update_l_kernel(ngen, vr_sol.l_curr, vr_sol.z_curr, vr_sol.lz, par.beta)
    end
    
    # TODO: add time update later
    return
end
