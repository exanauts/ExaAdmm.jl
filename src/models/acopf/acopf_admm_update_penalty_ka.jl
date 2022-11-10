@kernel function update_penalty_ka(nvar::Int, theta::Float64, gamma::Float64,
    rp_curr, rp_prev,
    rho
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))
    if tx <= nvar
        if rp_curr[tx] > theta*rp_prev[tx]
            rho[tx] *= gamma
        end
    end
end

function acopf_admm_update_penalty(
    env::AdmmEnv,
    mod::Model,
    device
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
