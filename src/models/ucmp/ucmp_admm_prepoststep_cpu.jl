function admm_outer_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    admm_outer_prestep(env, mod.mpmodel, device)
    
    # v ramp consensus constraints.
    vr_sols = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        vr_sol = vr_sols[i]
        model.info.norm_z_prev = sqrt(model.info.norm_z_prev^2 + norm(vr_sol.z_curr)^2)
    end

    # uc variable consensus constraints
    uc_sol = mod.uc_solution
    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        model.info.norm_z_prev = sqrt(model.info.norm_z_prev^2 + norm(uc_sol.z_curr[:, 3*i-2:3*i])^2)
    end

    # update info for the UCMPModel level
    info = mod.info
    info.norm_z_prev = 0.0
    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        info.norm_z_prev = max(info.norm_z_prev, model.info.norm_z_prev)
    end

    return
end

"""
Implement any algorithmic steps required before each inner iteration.
"""
function admm_inner_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    admm_inner_prestep(env, mod.mpmodel, device)
    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution
    uc_sol.z_prev .= uc_sol.z_curr
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        vr_sol.z_prev .= vr_sol.z_curr
    end
    return
end

"""
Implement any steps required after the algorithm terminates.
"""
function admm_poststep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    info = mod.info
    admm_poststep(env, mod.mpmodel, device) # NOTE: the calculation for info.user.err_ramp (ramp violation) in this step is wrong for UCMP, avoid using this
    info.objval = mod.mpmodel.info.objval
    uc_sol = mod.uc_solution
    on_cost, off_cost = mod.uc_params.con, mod.uc_params.coff
    # add uc objective terms
    for i=1:mod.mpmodel.len_horizon
        info.objval += sum(uc_sol.u_curr[:, 3*i-1] .* on_cost) + sum(uc_sol.u_curr[:, 3*i] .* off_cost)
    end
    return
end