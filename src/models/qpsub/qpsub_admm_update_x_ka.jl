"""
    acopf_admm_update_x_line()

- update xline: call auglag_linelimit_qpsub() = update sol.x[pij_idx]
- record run time info.user.time_branches, info.time_x_update
"""

function acopf_admm_update_x_line(
    env::AdmmEnv,
    mod::ModelQpsub,
    device
    )
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data

    ev = auglag_linelimit_qpsub(device, 32, data.nline*32)(
        mod.Hs, sol.l_curr, sol.rho, sol.u_curr, sol.v_curr, sol.z_curr, mod.grid_data.YffR, mod.grid_data.YffI,
        mod.grid_data.YftR, mod.grid_data.YftI,
        mod.grid_data.YttR, mod.grid_data.YttI,
        mod.grid_data.YtfR, mod.grid_data.YtfI, info.inner, par.max_auglag, par.mu_max, par.scale, mod.ls, mod.us, mod.sqp_line,
        mod.qpsub_membuf, mod.LH_1h, mod.RH_1h, mod.LH_1i, mod.RH_1i, mod.LH_1j, mod.RH_1j, mod.LH_1k, mod.RH_1k, mod.lambda, mod.line_start, mod.grid_data.nline, mod.supY, mod.line_res
    )
    KA.synchronize(device)

    info.user.time_branches += 0.0
    info.time_x_update += 0.0

return
end
