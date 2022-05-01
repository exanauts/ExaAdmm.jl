function mpacopf_update_z_kernel(
    n::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1}, lz::CuDeviceArray{Float64,1},
    beta::Float64
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        gid = gen_start + 2*(tx-1)
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(u[tx] - v[gid]))) / (beta + rho[tx])
        end
    end
    return
end

function admm_update_z(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_z(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i-1], mod.solution[i]
        ztime = CUDA.@timed @cuda threads=64 blocks=(div(submod.ngen-1, 64)+1) mpacopf_update_z_kernel(
            submod.ngen, submod.gen_start, sol_ramp.u_curr, submod.solution.v_curr, sol_ramp.z_curr,
            sol_ramp.l_curr, sol_ramp.rho, sol_ramp.lz, par.beta
        )
        submod.info.time_z_update += ztime.time
    end
    return
end