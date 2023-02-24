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
    
- update xline: call auglag_linelimit_qpsub() = update sol.x[pij_idx]
- record run time info.user.time_branches, info.time_x_update
"""

function admm_update_x_line(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
    )
    par, sol, info, data = env.params, mod.solution, mod.info, mod.grid_data

    shmem_size = env.params.shmem_size

    time_br = CUDA.@timed @cuda threads=32 blocks=data.nline shmem=shmem_size auglag_linelimit_qpsub(mod.Hs, sol.l_curr, sol.rho, sol.u_curr, sol.v_curr, sol.z_curr, mod.grid_data.YffR, mod.grid_data.YffI,
    mod.grid_data.YftR, mod.grid_data.YftI,
    mod.grid_data.YttR, mod.grid_data.YttI,
    mod.grid_data.YtfR, mod.grid_data.YtfI, info.inner, par.max_auglag, par.mu_max, par.scale, mod.ls, mod.us, mod.sqp_line,
    mod.qpsub_membuf, mod.LH_1h, mod.RH_1h, mod.LH_1i, mod.RH_1i, mod.LH_1j, mod.RH_1j, mod.LH_1k, mod.RH_1k, mod.lambda, mod.line_start, mod.grid_data.nline, mod.supY, mod.line_res)

    info.user.time_branches += time_br.time
    info.time_x_update += time_br.time
    
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
    admm_update_x_line(env, mod) 


    return
end