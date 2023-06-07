function acopf_admm_update_x_gen(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    gen_solution::EmptyGeneratorSolution,
    device
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    generator_kernel_two_level(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho, device)
    info.user.time_generators += 0.0
    info.time_x_update += 0.0
end

function acopf_admm_update_x_line(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device
)
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data
    if env.use_linelimit
        auglag_linelimit_two_level_alternative_ka(device, 32, 32*data.nline)(
            Val(mod.n), data.nline, mod.line_start,
            info.inner, par.max_auglag, par.mu_max, par.scale,
            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            par.shift_lines, mod.membuf, data.YffR, data.YffI, data.YftR, data.YftI,
            data.YttR, data.YttI, data.YtfR, data.YtfI,
            data.FrVmBound, data.ToVmBound, data.FrVaBound, data.ToVaBound
        )
        KA.synchronize(device)
    else
        polar_kernel_two_level_alternative_ka(device, 32, 32*data.nline)(
            mod.n, data.nline, mod.line_start, par.scale,
            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            par.shift_lines, mod.membuf, data.YffR, data.YffI, data.YftR, data.YftI,
            data.YttR, data.YttI, data.YtfR, data.YtfI, data.FrVmBound, data.ToVmBound
        )
        KA.synchronize(device)
    end
    info.time_x_update += 0.0
    info.user.time_branches += 0.0
    return
end

function admm_update_x(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device
)
    acopf_admm_update_x_gen(env, mod, mod.gen_solution, device)
    # acopf_admm_update_x_line(env, mod, device)
    return
end
