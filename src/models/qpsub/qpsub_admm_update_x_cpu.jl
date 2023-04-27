"""
    acopf_admm_update_x_line()

- update xline: call auglag_linelimit_two_level_alternative_qpsub() = update sol.x[pij_idx]
- record run time info.user.time_branches, info.time_x_update
"""

function acopf_admm_update_x_line(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
    )
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data

@inbounds begin
    for i = 1 : mod.grid_data.nline
        shift_idx = mod.line_start + 8*(i-1)
        A_ipopt = eval_A_branch_kernel_cpu_qpsub(mod.Hs[6*(i-1)+1:6*i,1:6], sol.l_curr[shift_idx : shift_idx + 7],
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7],
        sol.z_curr[shift_idx : shift_idx + 7],
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])

        b_ipopt = eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx : shift_idx + 7],
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7],
        sol.z_curr[shift_idx : shift_idx + 7],
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i], mod.line_res[:,i])

        #call reduced branch kernel
        time_br = @timed tronx, tronf = auglag_Ab_linelimit_two_level_alternative_qpsub_ij_red(info.inner, par.max_auglag, par.mu_max, par.scale, A_ipopt, b_ipopt, mod.ls[i,:], mod.us[i,:], mod.sqp_line, sol.l_curr[shift_idx : shift_idx + 7],
        sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7],
        sol.z_curr[shift_idx : shift_idx + 7], mod.qpsub_membuf,i,
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],
        mod.LH_1h[i,:], mod.RH_1h[i], mod.LH_1i[i,:], mod.RH_1i[i], mod.LH_1j[i,:], mod.RH_1j[i], mod.LH_1k[i,:], mod.RH_1k[i], mod.lambda, mod.line_res[:,i])



    info.user.time_branches += time_br.time
    info.time_x_update += time_br.time
    end
end #@inbounds

return
end
