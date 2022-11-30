"""
Compute and update multipliers `λ_z` for the augmented Lagrangian wit respect to `z=0` constraint.
"""
function admm_update_lz(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    admm_update_lz(env, mod.mpmodel, device)

    par = env.params
    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution
    # update lz for uc binary solution
    uc_sol.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, uc_sol.lz .+ (par.beta .* uc_sol.z_curr)))

    # update lz for v ramp solution
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        vr_sol.lz .= max.(-par.MAX_MULTIPLIER, min.(par.MAX_MULTIPLIER, vr_sol.lz .+ (par.beta .* vr_sol.z_curr)))
    end
    return
end