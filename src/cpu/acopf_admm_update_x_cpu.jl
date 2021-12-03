function acopf_admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation{ComponentInformation}
)
    par, sol = env.params, mod.solution

    time_gen = generator_kernel_two_level(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)

    if env.use_linelimit
        time_br = @timed auglag_it, tron_it = auglag_linelimit_two_level_alternative(mod.n, mod.nline, mod.line_start,
                                                info.inner, par.max_auglag, par.mu_max, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI,
                                                mod.FrVmBound, mod.ToVmBound, mod.FrVaBound, mod.ToVaBound)
    else
        time_br = @timed auglag_it, tron_it = polar_kernel_two_level_alternative(mod.n, mod.nline, mod.line_start, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrVmBound, mod.ToVmBound)
    end

    info.time_x_update += time_gen.time + time_br.time
    info.user.time_generators += time_gen.time
    info.user.time_branches += time_br.time

    return
end