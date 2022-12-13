"""
Update variable `z`, representing the artificial variables that are driven to zero in the two-level ADMM.
"""
function admm_update_z(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    admm_update_z(env, mod.mpmodel, device)
    # update z for binary variables
    par = env.params
    uc_sol = mod.uc_solution
    uc_sol.z_curr .= (-(uc_sol.lz .+ uc_sol.l_curr .+ uc_sol.rho .* (uc_sol.u_curr .- uc_sol.v_curr))) ./ (par.beta .+ uc_sol.rho)

    # update z for v ramp variables
    vr_sols = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        v_curr = @view uc_sol.v_curr[:, 3*i-5]
        vr_sol.z_curr .= (-(vr_sol.lz .+ vr_sol.l_curr .+ vr_sol.rho .* (vr_sol.u_curr .- v_curr))) ./ (par.beta .+ vr_sol.rho)
    end

    # TODO: add time update later
    return
end
