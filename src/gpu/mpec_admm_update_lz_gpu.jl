function acopf_admm_update_lz(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info
    lztime = CUDA.@timed @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, par.beta)
    info.time_lz_update += lztime.time
    return
end