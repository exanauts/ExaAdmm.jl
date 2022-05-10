function acopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    gen_sol::EmptyGeneratorSolution{Float64,Array{Float64,1}}
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    time_gen = generator_kernel_two_level_qpsub(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time
    return
end

function acopf_admm_update_x_line(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data

#=
    tmp = mod.nline
    mod.nline = 1
    par.shift_lines = 3990
=#
    if env.use_linelimit
        time_br = @timed auglag_it, tron_it = auglag_linelimit_two_level_alternative_qpsub(mod.n, data.nline, mod.line_start,
                                                info.inner, par.max_auglag, par.mu_max, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, data.YffR, data.YffI, data.YftR, data.YftI,
                                                data.YttR, data.YttI, data.YtfR, data.YtfI,
                                                data.FrVmBound, data.ToVmBound, data.FrVaBound, data.ToVaBound)
    else
        time_br = @timed auglag_it, tron_it = polar_kernel_two_level_alternative(mod.n, data.nline, mod.line_start, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, data.YffR, data.YffI, data.YftR, data.YftI,
                                                data.YttR, data.YttI, data.YtfR, data.YtfI, data.FrVmBound, data.ToVmBound)
    end
#    mod.nline = tmp

    info.user.time_branches += time_br.time
    info.time_x_update += time_br.time
    return
end

function admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    acopf_admm_update_x_gen(env, mod, mod.gen_solution)
    acopf_admm_update_x_line(env, mod)
    return
end