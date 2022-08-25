function admm_update_lz(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_lz(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i], mod.solution[i]
        lztime = @timed sol_ramp.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, sol_ramp.lz .+ (par.beta .* sol_ramp.z_curr)))
        submod.info.time_lz_update += lztime.time
    end
    return
end