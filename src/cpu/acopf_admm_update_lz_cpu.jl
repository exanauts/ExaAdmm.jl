function acopf_admm_update_lz(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation
)
    par, sol = env.params, mod.solution
    lztime = @timed sol.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, sol.lz .+ (par.beta .* sol.z_curr)))
    info.time_lz_update += lztime.time
    return
end