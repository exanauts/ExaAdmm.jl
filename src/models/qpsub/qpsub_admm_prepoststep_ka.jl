"""
    admm_poststep()

- after admm termination, fix solution pf_projection() and record time mod.info.time_projection
- update info.objval and info.auglag with projected solution
- feed step solution, multipliers and KKT error info to SQOPF
"""

function admm_poststep(
    env::AdmmEnv,
    mod::ModelQpsub,
    device
)
    par, data, sol, info, grid_data = env.params, env.data, mod.solution, mod.info, mod.grid_data

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    u_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)

    l_curr = Array(sol.l_curr)
    rho = Array(sol.rho)
    rp = Array(sol.rp)

    c2 = zeros(grid_data.ngen)
    copyto!(c2, mod.qpsub_c2)

    c1 = zeros(grid_data.ngen)
    copyto!(c1, mod.qpsub_c1)

    sqp_line = Array(mod.sqp_line)

    FrStart = Array(grid_data.FrStart)
    FrIdx = Array(grid_data.FrIdx)

    ToStart = Array(grid_data.ToStart)
    ToIdx = Array(grid_data.ToIdx)

    Hs = Array(mod.Hs)

    #final objective and auglag
    info.objval = sum(c2[g]*(grid_data.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                    c1[g]*(grid_data.baseMVA*u_curr[mod.gen_start+2*(g-1)])
                    for g in 1:grid_data.ngen) +
                        sum(0.5*dot(sqp_line[:,l],Hs[6*(l-1)+1:6*l,1:6],sqp_line[:,l]) for l=1:grid_data.nline)

    info.auglag = info.objval
    info.auglag += sum(l_curr[i]*rp[i] for i=1:mod.nvar)
    info.auglag += 0.5*sum(rho[i]*(rp[i])^2 for i=1:mod.nvar)


    #find dual infeas kkt for SQP integration
    pg_dual_infeas = [2*c2[g]*(grid_data.baseMVA)^2*u_curr[mod.gen_start+2*(g-1)] for g = 1:grid_data.ngen]
    line_dual_infeas = vcat([Hs[6*(l-1)+1:6*l,1:6] * sqp_line[:,l] for l = 1:grid_data.nline]...)
    mod.dual_infeas = vcat(pg_dual_infeas, line_dual_infeas) #unscale

    #assign value to step variable; due to inexact steps, other options include use u_curr, v_curr alone or average
    #generation
    @inbounds begin
        for g = 1: grid_data.ngen
            pg_idx = mod.gen_start + 2*(g-1)
            qg_idx = mod.gen_start + 2*(g-1) + 1
            mod.dpg_sol[g] = u_curr[pg_idx]
            mod.dqg_sol[g] = u_curr[qg_idx]
        end

        copyto!(mod.dline_var, sqp_line)

        for l = 1:grid_data.nline
            shift_idx = mod.line_start + 8*(l-1)
            mod.dline_fl[1,l] = u_curr[shift_idx] #pij
            mod.dline_fl[2,l] = u_curr[shift_idx + 1] #qij
            mod.dline_fl[3,l] = u_curr[shift_idx + 2] #pji
            mod.dline_fl[4,l] = u_curr[shift_idx + 3] #qji
        end

        for b = 1: grid_data.nbus
            dw_ct = 0
            dw_sum = 0
            dt_sum = 0
            dt_ct = 0
            if FrStart[b] < FrStart[b+1]
                for k = FrStart[b]:FrStart[b+1]-1
                    dw_sum  += sqp_line[3, FrIdx[k]] #wi(ij)
                    dw_ct += 1
                    dt_sum += sqp_line[5, FrIdx[k]] #ti(ij)
                    dt_ct += 1
                end
            end
            if ToStart[b] < ToStart[b+1]
                for k = ToStart[b]:ToStart[b+1]-1
                    dw_sum += sqp_line[4, ToIdx[k]] #wj(ji)
                    dt_sum += sqp_line[6, ToIdx[k]] #tj(ji)
                    dw_ct += 1
                    dt_ct += 1
                end
            end
            mod.dw_sol[b] = dw_sum/dw_ct #average
            mod.dtheta_sol[b] = dt_sum/dt_ct #average
        end
    end #inbound
    return
end
