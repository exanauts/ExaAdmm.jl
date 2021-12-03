function acopf_admm_outer_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation{ComponentInformation}
)
    sol = mod.solution
    info.norm_z_prev = norm(sol.z_curr)
    return
end

function acopf_admm_inner_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation{ComponentInformation}
)
    sol = mod.solution
    sol.z_prev .= sol.z_curr
    return
end

function acopf_admm_poststep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation{ComponentInformation}
)
    data, sol = env.data, mod.solution

    info.user.err_pg, info.user.err_qg = check_generator_bounds(mod, sol.u_curr)
    info.user.err_vm = check_voltage_bounds_alternative(mod, sol.v_curr)
    info.user.err_real, info.user.err_reactive = check_power_balance_alternative(mod, sol.u_curr, sol.v_curr)
    info.user.num_rateA_viols, info.user.err_rateA = check_linelimit_violation(data, sol.v_curr)

    sol.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                    data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)]) +
                    data.generators[g].coeff[data.generators[g].n]
                    for g in 1:mod.ngen)::Float64
    #=
    sol.cumul_iters = info.cumul
    sol.overall_time = info.time_overall
    sol.status = (info.mismatch <= OUTER_TOL) ? :Solved : :IterLimit
    sol.max_viol_except_line = max(info.user.err_pg, info.user.err_qg, info.user.err_vm,
                                   info.user.err_real, info.user.err_reactive)
    sol.max_line_viol_rateA = info.user.err_rateA
    =#
end