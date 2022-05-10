function admm_outer_prestep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol_ramp, info = mod.solution, mod.info

    for i=1:mod.len_horizon
        admm_outer_prestep(env, mod.models[i])
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

function admm_inner_prestep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon
        admm_inner_prestep(env, mod.models[i])
    end

    # Ramp related consensus constraints.
    sol_ramp = mod.solution
    for i=2:mod.len_horizon
        sol_ramp[i].z_prev .= sol_ramp[i].z_curr
    end
    return
end

function check_ramp_violations(mod::ModelAcopf, u_curr::Vector{Float64}, u_prev::Vector{Float64}, ramp_rate::Vector{Float64})
    max_viol = 0.0
    for g=1:mod.grid_data.ngen
        pg_idx = mod.gen_start + 2*(g-1)
        max_viol = max(max_viol, max(0.0, -(ramp_rate[g]-abs(u_curr[pg_idx]-u_prev[pg_idx]))))
    end
    return max_viol
end

function admm_poststep(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon
        admm_poststep(env, mod.models[i])
    end
    for i=2:mod.len_horizon
        mod.models[i].info.user.err_ramp = check_ramp_violations(mod.models[i], mod.models[i].solution.u_curr, mod.models[i-1].solution.u_curr, mod.models[i].grid_data.ramp_rate)
    end

    info = mod.info
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
end