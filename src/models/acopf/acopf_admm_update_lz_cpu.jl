"""
Compute and update multipliers `λ_z` for the augmented Lagrangian wit respect to `z=0` constraint.
"""
function admm_update_lz(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info
    lztime = @timed sol.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, sol.lz .+ (par.beta .* sol.z_curr)))
    info.time_lz_update += lztime.time
    return
end
