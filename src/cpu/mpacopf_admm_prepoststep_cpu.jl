function acopf_admm_outer_prestep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol_ramp, info = mod.solution, mod.info

    for i=1:mod.len_horizon
        acopf_admm_outer_prestep(env, mod.models[i])
    end
    # Ramp related consensus constraints.
    for i=2:mod.len_horizon
        mod.models[i].info.norm_z_prev = sqrt(mod.models[i].info.norm_z_prev^2 + norm(sol_ramp[i].z_curr)^2)
    end
    info.norm_z_prev = 0.0
    for i=1:mod.len_horizon
        info.norm_z_prev = max(info.norm_z_prev, mod.models[i].info.norm_z_prev)
    end
    return
end

function acopf_admm_inner_prestep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon
        acopf_admm_inner_prestep(env, mod.models[i])
    end

    # Ramp related consensus constraints.
    sol_ramp = mod.solution
    for i=2:mod.len_horizon
        sol_ramp[i].z_prev .= sol_ramp[i].z_curr
    end
    return
end


function acopf_admm_poststep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon
        acopf_admm_poststep(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        mod.models[i].info.user.err_ramp = check_ramp_violations(mod.models[i], mod.models[i].solution.u_curr, mod.models[i-1].solution.u_curr, mod.models[i].ramp_rate)
    end

    info = mod.info
    info.user.err_pg = maximum([mod.models[i].info.user.err_pg for i=1:mod.len_horizon])
    info.user.err_qg = maximum([mod.models[i].info.user.err_qg for i=1:mod.len_horizon])
    info.user.err_vm = maximum([mod.models[i].info.user.err_vm for i=1:mod.len_horizon])
    info.user.err_real = maximum([mod.models[i].info.user.err_real for i=1:mod.len_horizon])
    info.user.err_reactive = maximum([mod.models[i].info.user.err_reactive for i=1:mod.len_horizon])
    info.user.err_rateA = maximum([mod.models[i].info.user.err_rateA for i=1:mod.len_horizon])
    info.user.num_rateA_viols = sum(mod.models[i].info.user.num_rateA_viols for i=1:mod.len_horizon)
    if mod.len_horizon > 1
        info.user.err_ramp = maximum([mod.models[i].info.user.err_ramp for i=2:mod.len_horizon])
    else
        info.user.err_ramp = 0.0
    end
    info.user.time_generators = sum(mod.models[i].info.user.time_generators for i=1:mod.len_horizon)
    info.user.time_branches = sum(mod.models[i].info.user.time_branches for i=1:mod.len_horizon)
    info.user.time_buses = sum(mod.models[i].info.user.time_buses for i=1:mod.len_horizon)
    info.time_x_update = sum(mod.models[i].info.time_x_update for i=1:mod.len_horizon)
    info.time_xbar_update = sum(mod.models[i].info.time_xbar_update for i=1:mod.len_horizon)
    info.time_z_update = sum(mod.models[i].info.time_z_update for i=1:mod.len_horizon)
    info.time_l_update = sum(mod.models[i].info.time_l_update for i=1:mod.len_horizon)
    info.time_lz_update = sum(mod.models[i].info.time_lz_update for i=1:mod.len_horizon)
    info.objval = sum(mod.models[i].info.objval for i=1:mod.len_horizon)

    #=
    sol.cumul_iters = info.cumul
    sol.overall_time = info.time_overall
    sol.status = (info.mismatch <= OUTER_TOL) ? :Solved : :IterLimit
    sol.max_viol_except_line = max(info.user.err_pg, info.user.err_qg, info.user.err_vm,
                                    info.user.err_real, info.user.err_reactive)
    sol.max_line_viol_rateA = info.user.err_rateA
    =#
end