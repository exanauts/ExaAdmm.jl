"""
    update_l_kernel_single
"""
function update_l_kernel_single(
    n::Int, l_curr::CuDeviceArray{Float64,1}, l_prev::CuDeviceArray{Float64,1}, u::CuDeviceArray{Float64,1},
    v::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1}
    )
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            l_curr[tx] = l_prev[tx] + rho[tx] * (u[tx] - v[tx])
        end
    end

    return
end

"""
    admm_update_l()
    
- update l
- record time info.time_l_update
"""

function admm_update_l_single(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
    )
    par, sol, info = env.params, mod.solution, mod.info
    sol.l_prev = sol.l_curr
    ltime = CUDA.@timed @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_l_kernel_single(mod.nvar, sol.l_curr, sol.l_prev, sol.u_curr, sol.v_curr, sol.rho)
    info.time_l_update += ltime.time
    return
end