function admm_update_l(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_l(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i], mod.solution[i]
        ltime = @timed sol_ramp.l_curr .= -(sol_ramp.lz .+ par.beta .* sol_ramp.z_curr)
        submod.info.time_l_update += ltime.time
    end
    return
end