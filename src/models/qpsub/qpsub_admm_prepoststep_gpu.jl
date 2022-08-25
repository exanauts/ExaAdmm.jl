"""
    admm_outer_prestep()
    
at start of each outer loop, preset info.norm_z_prev
"""

function admm_outer_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = CUDA.norm(sol.z_curr)
    return
end





"""
    admm_inner_prestep()
    
at start of each inner loop, preset sol.z_prev
"""

function admm_inner_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol = mod.solution
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.z_prev, sol.z_curr)
    CUDA.synchronize()
    return
end





"""
    admm_poststep()
    
- after admm termination, fix solution pf_projection() and record time mod.info.time_projection
- update info.objval with projected solution
"""

function admm_poststep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, data, sol, info, grid_data = env.params, env.data, mod.solution, mod.info, mod.grid_data

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    #final objective and auglag 
    info.objval = sum(mod.qpsub_c2[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                    mod.qpsub_c1[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])
                    for g in 1:grid_data.ngen) + 
                        sum(0.5*dot(mod.sqp_line[:,l],mod.Hs[6*(l-1)+1:6*l,1:6],mod.sqp_line[:,l]) for l=1:grid_data.nline) 
    
    info.auglag = info.objval + sum(sol.lz[i]*sol.z_curr[i] for i=1:mod.nvar) +
                        0.5*par.beta*sum(sol.z_curr[i]^2 for i=1:mod.nvar) +
                        sum(sol.l_curr[i]*sol.rp[i] for i=1:mod.nvar) +
                        0.5*sum(sol.rho[i]*(sol.rp[i])^2 for i=1:mod.nvar)   


    #find dual infeas kkt for SQP integration
    pg_dual_infeas = [2*mod.qpsub_c2[g]*(grid_data.baseMVA)^2*sol.u_curr[mod.gen_start+2*(g-1)] for g = 1:grid_data.ngen]
    line_dual_infeas = vcat([mod.Hs[6*(l-1)+1:6*l,1:6] * mod.sqp_line[:,l] for l = 1:grid_data.nline]...)
    mod.dual_infeas = vcat(pg_dual_infeas, line_dual_infeas) #unscale 

    #assign value to step variable
    #generation
    @inbounds begin
        for g = 1: grid_data.ngen
            pg_idx = mod.gen_start + 2*(g-1)
            qg_idx = mod.gen_start + 2*(g-1) + 1
            mod.dpg_sol[g] = sol.u_curr[pg_idx] #? use u+v/2
            mod.dqg_sol[g] = sol.u_curr[qg_idx]
        end
        
        mod.dline_var = copy(mod.sqp_line)

        for l = 1:grid_data.nline
            shift_idx = mod.line_start + 8*(l-1)
            mod.dline_fl[1,l] = sol.u_curr[shift_idx] #pij
            mod.dline_fl[2,l] = sol.u_curr[shift_idx + 1] #qij
            mod.dline_fl[3,l] = sol.u_curr[shift_idx + 2] #pji
            mod.dline_fl[4,l] = sol.u_curr[shift_idx + 3] #qji
        end

        for b = 1: grid_data.nbus
            dw_ct = 0
            dw_sum = 0
            dt_sum = 0
            dt_ct = 0
            if grid_data.FrStart[b] < grid_data.FrStart[b+1]
                for k = grid_data.FrStart[b]:grid_data.FrStart[b+1]-1
                    dw_sum  += mod.dline_var[3, grid_data.FrIdx[k]] #wi(ij)
                    dw_ct += 1
                    dt_sum += mod.dline_var[5, grid_data.FrIdx[k]] #ti(ij)
                    dt_ct += 1
                end
            end
            if grid_data.ToStart[b] < grid_data.ToStart[b+1]
                for k = grid_data.ToStart[b]:grid_data.ToStart[b+1]-1
                    dw_sum += mod.dline_var[4, grid_data.ToIdx[k]] #wj(ji)
                    dt_sum += mod.dline_var[6, grid_data.ToIdx[k]] #tj(ji)
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