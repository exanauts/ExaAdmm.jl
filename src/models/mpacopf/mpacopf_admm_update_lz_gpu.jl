function admm_update_lz(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    par = env.params

    for i=1:mod.len_horizon
        admm_update_lz(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i], mod.solution[i]
        lztime = CUDA.@timed @cuda threads=64 blocks=(div(submod.grid_data.ngen-1, 64)+1) update_lz_kernel(submod.grid_data.ngen, par.MAX_MULTIPLIER, sol_ramp.z_curr, sol_ramp.lz, par.beta)
        submod.info.time_lz_update += lztime.time
    end
    return
end