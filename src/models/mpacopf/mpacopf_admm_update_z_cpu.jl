function admm_update_z(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_z(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i-1], mod.solution[i]
        v_curr = @view submod.solution.v_curr[submod.gen_start:2:submod.gen_start+2*submod.grid_data.ngen-1]
        ztime = @timed sol_ramp.z_curr .= (-(sol_ramp.lz .+ sol_ramp.l_curr .+ sol_ramp.rho .* (sol_ramp.u_curr .- v_curr))) ./ (par.beta .+ sol_ramp.rho)
        submod.info.time_z_update += ztime.time
    end
    return
end