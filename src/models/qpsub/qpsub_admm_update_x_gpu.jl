"""
    admm_update_x_gen()
    
- update xgen: call generator_kernel_two_level_qpsub() = update sol.x[pg_idx], sol.x[qp_idx]
- record run time info.user.time_generators, info.time_x_update
"""

function admm_update_x_gen(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    gen_solution::EmptyGeneratorSolution{Float64,CuArray{Float64,1}}
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    time_gen = generator_kernel_two_level_qpsub(mod, data.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time
    return
end


"""
    admm_update_x_line()
    
- update xline: call auglag_linelimit_two_level_alternative_qpsub() = update sol.x[pij_idx]
- record run time info.user.time_branches, info.time_x_update
"""

function admm_update_x_line(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
    )
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data

    # nblk = div(model.grid_data.ngen, 32, RoundUp)
    shmem_size = env.params.shmem_size

    #reset
    fill!(mod.A_ipopt, 0.0)
    fill!(mod.b_ipopt, 0.0)

    #use nline blocks 
    time_br = CUDA.@timed @cuda threads=32 blocks=data.nline shmem=shmem_size auglag_linelimit_qpsub(mod.Hs, sol.l_curr, sol.rho, sol.u_curr, sol.v_curr, sol.z_curr, mod.grid_data.YffR, mod.grid_data.YffI,
    mod.grid_data.YftR, mod.grid_data.YftI,
    mod.grid_data.YttR, mod.grid_data.YttI,
    mod.grid_data.YtfR, mod.grid_data.YtfI, info.inner, par.max_auglag, par.mu_max, par.scale, mod.ls, mod.us, mod.sqp_line,
    mod.qpsub_membuf, mod.LH_1h, mod.RH_1h, mod.LH_1i, mod.RH_1i, mod.LH_1j, mod.RH_1j, mod.LH_1k, mod.RH_1k, mod.lambda, mod.line_start, mod.grid_data.nline, mod.A_ipopt, mod.b_ipopt, mod.supY)

    info.user.time_branches += time_br.time
    info.time_x_update += time_br.time
    
return
end

function auglag_linelimit_qpsub(Hs, l_curr, rho, u_curr, v_curr, z_curr, YffR, YffI,
         YftR, YftI, YttR, YttI, YtfR, YtfI, inner, max_auglag, mu_max, scale, ls, us, sqp_line,
         qpsub_membuf, LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k, RH_1k, lambda, line_start, nline, A_ipopt, b_ipopt, supY)

    tx = threadIdx().x
    id_line = blockIdx().x
    # id_line = I + shift_lines
    shift_idx = line_start + 8*(id_line-1)

    if id_line <= nline 
        #  eval_A_branch_kernel_gpu_qpsub(Hs[6*(id_line - 1) + 1:6*id_line, 1:6], l_curr[shift_idx : shift_idx + 7], 
        #     rho[shift_idx : shift_idx + 7], v_curr[shift_idx : shift_idx + 7], 
        #     z_curr[shift_idx : shift_idx + 7], 
        #     YffR[id_line], YffI[id_line],
        #     YftR[id_line], YftI[id_line],
        #     YttR[id_line], YttI[id_line],
        #     YtfR[id_line], YtfI[id_line], A_ipopt)

        eval_A_b_branch_kernel_gpu_qpsub(Hs, l_curr, rho, v_curr, z_curr, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, A_ipopt, b_ipopt, id_line, shift_idx, supY)
    
            # print(mod.Hs[6*(i-1)+1:6*i,1:6])
    
        # b_ipopt = eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx : shift_idx + 7], 
        #     sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        #     sol.z_curr[shift_idx : shift_idx + 7], 
        #     mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        #     mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        #     mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        #     mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])
    
            # println(A_ipopt)
            # println(b_ipopt)
            
            #ij or ij_red
        # tronx, tronf = auglag_Ab_linelimit_two_level_alternative_qpsub_ij_red(info.inner, par.max_auglag, par.mu_max, par.scale, A_ipopt, b_ipopt, mod.ls[i,:], mod.us[i,:], mod.sqp_line, sol.l_curr[shift_idx : shift_idx + 7], 
        #     sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7], 
        #     sol.z_curr[shift_idx : shift_idx + 7], mod.qpsub_membuf,i,
        #     mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        #     mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        #     mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        #     mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],
        #     mod.LH_1h[i,:], mod.RH_1h[i], mod.LH_1i[i,:], mod.RH_1i[i], mod.LH_1j[i,:], mod.RH_1j[i], mod.LH_1k[i,:], mod.RH_1k[i], mod.lambda)
    
    
            # println(mod.solution.u_curr)
    end #if


    return
end



"""
    admm_update_x()
    
update xgen and xline
"""

function admm_update_x(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    admm_update_x_gen(env, mod, mod.gen_solution)

    # println(mod.solution.u_curr)
    admm_update_x_line(env, mod) 


    return
end