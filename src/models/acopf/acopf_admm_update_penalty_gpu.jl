function update_penalty(nvar::Int, theta::Float64, gamma::Float64,
    rp_curr::CuDeviceArray{Float64,1}, rp_prev::CuDeviceArray{Float64,1},
    rho::CuDeviceArray{Float64,1}
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= nvar
        if rp_curr[tx] > theta*rp_prev[tx]
            rho[tx] *= gamma
        end
    end
    return
end

function acopf_admm_update_penalty(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
#=
    par, sol = env.params, mod.solution
    norm_rp_prev = CUDA.norm(sol.rp_prev)
    norm_rp_curr = CUDA.norm(sol.rp)
    if norm_rp_curr > 0.98*norm_rp_prev
        sol.rho .= min.(par.rho_max, 2 .* sol.rho)
    end

    par, sol = env.params, mod.solution
    if (mod.info.inner % 100) == 0
        CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1,64)+1) update_penalty(mod.nvar, par.theta, par.inc_c, sol.rp, sol.rp_prev, sol.rho)
    end
=#
    return
end