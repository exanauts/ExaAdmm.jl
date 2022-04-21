function admm_outer_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = norm(sol.z_curr)
    return
end

function admm_inner_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol = mod.solution
    sol.z_prev .= sol.z_curr
    return
end

function admm_poststep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    data, sol, info = env.data, mod.solution, mod.info

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    info.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                      data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)]) +
                      data.generators[g].coeff[data.generators[g].n]
                      for g in 1:mod.ngen)::Float64
end