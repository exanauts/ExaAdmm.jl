"""
Update multipliers λ for consensus constraints, `x - xbar + z = 0`.
"""
function admm_update_l(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    admm_update_l(env, mod.mpmodel, device)

    # update l for uc_solution
    par = env.params
    uc_sol = mod.uc_solution
    uc_sol.l_curr .= -(uc_sol.lz .+ par.beta .* uc_sol.z_curr)

    # update l for v ramp solution
    vr_sols = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        vr_sol.l_curr .= -(vr_sol.lz .+ par.beta .* vr_sol.z_curr)
    end
    
    # TODO: add time update later
    return
end
