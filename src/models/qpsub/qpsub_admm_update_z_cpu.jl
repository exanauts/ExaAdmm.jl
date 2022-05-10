function admm_update_z(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info
    ztime = @timed sol.z_curr .= (-(sol.lz .+ sol.l_curr .+ sol.rho.*(sol.u_curr .- sol.v_curr))) ./ (par.beta .+ sol.rho)
    info.time_z_update += ztime.time
    return
end