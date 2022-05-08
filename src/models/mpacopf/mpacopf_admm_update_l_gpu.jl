function admm_update_l(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_l(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i], mod.solution[i]
        ltime = CUDA.@timed @cuda threads=64 blocks=(div(submod.grid_data.ngen-1, 64)+1) update_l_kernel(submod.grid_data.ngen, sol_ramp.l_curr, sol_ramp.z_curr, sol_ramp.lz, par.beta)
        submod.info.time_l_update += ltime.time
    end
    return
end