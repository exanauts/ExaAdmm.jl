"""
    admm_update_z()

- update sol.z_curr for all coupling
- record run time info.time_z_update
- only called in two-level admm  
"""

function update_zv_kernel(n::Int, u, v, z, l, rho, lz, beta)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(u[tx] - v[tx]))) / (beta + rho[tx])
        end
    end

    return
end

function admm_update_z(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info

    ztime = CUDA.@timed @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_zv_kernel(mod.nvar, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho, sol.lz, par.beta)
    info.time_z_update += ztime.time
    return
end