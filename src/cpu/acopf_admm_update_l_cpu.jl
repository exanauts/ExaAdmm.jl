function acopf_admm_update_l(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info
    ltime = @timed sol.l_curr .= -(sol.lz .+ par.beta.*sol.z_curr)
    info.time_l_update += ltime.time
    return
end