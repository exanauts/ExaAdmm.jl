"""
Update variable `z`, representing the artificial variables that are driven to zero in the two-level ADMM.
"""
function ucmp_update_uc_z_kernel(
    n::Int, T::Int,
    u::CuDeviceArray{Float64,2}, v::CuDeviceArray{Float64,2},
    z::CuDeviceArray{Float64,2}, l::CuDeviceArray{Float64,2},
    rho::CuDeviceArray{Float64,2}, lz::CuDeviceArray{Float64,2},
    beta::Float64
)
    bx = blockIdx().x
    tx = threadIdx().x
    if bx <= n && tx <= T 
        @inbounds begin
            z[bx,tx] = (-(lz[bx,tx] + l[bx,tx] + rho[bx,tx]*(u[bx,tx] - v[bx,tx]))) / (beta + rho[bx,tx])
        end
    end
    return
end

function ucmp_update_vr_z_kernel(
    n::Int, t::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,2},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1}, lz::CuDeviceArray{Float64,1},
    beta::Float64
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(u[tx] - v[tx, 3*t-5]))) / (beta + rho[tx])
        end
    end
    return
end

function admm_update_z(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    admm_update_z(env, mod.mpmodel, device)
    par = env.params
    uc_sol = mod.uc_solution
    ngen = mod.mpmodel.models[1].grid_data.ngen

    # update z for binary variables
    nthreads = 2^ceil(Int, log2(3*mod.mpmodel.len_horizon))
    ztime = CUDA.@timed @cuda threads=nthreads blocks=ngen ucmp_update_uc_z_kernel(
        ngen, 3*mod.mpmodel.len_horizon, uc_sol.u_curr, uc_sol.v_curr, uc_sol.z_curr,
        uc_sol.l_curr, uc_sol.rho, uc_sol.lz, par.beta
    )

    # update z for v ramp variables
    vr_sols = mod.vr_solution    
    for i=2:mod.mpmodel.len_horizon
        submod, vr_sol = mod.mpmodel.models[i-1], vr_sols[i]
        ztime = CUDA.@timed @cuda threads=64 blocks=(div(ngen-1, 64)+1) ucmp_update_vr_z_kernel(
            ngen, i, vr_sol.u_curr, uc_sol.v_curr, vr_sol.z_curr,
            vr_sol.l_curr, vr_sol.rho, vr_sol.lz, par.beta
        )
        submod.info.time_z_update += ztime.time
    end
    return
end
