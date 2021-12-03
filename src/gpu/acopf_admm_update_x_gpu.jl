function acopf_admm_update_x(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    info::IterationInformation{ComponentInformation}
)
    par, sol = env.params, mod.solution
    shmem_size = env.params.shmem_size
    time_gen = generator_kernel_two_level(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)

    if env.use_linelimit
        time_br = CUDA.@timed @cuda threads=32 blocks=mod.nline shmem=shmem_size auglag_linelimit_two_level_alternative(
                                            mod.n, mod.nline, mod.line_start,
                                            info.inner, par.max_auglag, par.mu_max, par.scale,
                                            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                            par.shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                            mod.YttR, mod.YttI, mod.YtfR, mod.YtfI,
                                            mod.FrVmBound, mod.ToVmBound, mod.FrVaBound, mod.ToVaBound)
    else
        time_br = CUDA.@timed @cuda threads=32 blocks=mode.nline shmem=shmem_size polar_kernel_two_level_alternative(mod.n, mod.nline, mod.line_start, par.scale,
                                                        sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                        par.shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                        mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrVmBound, mod.ToVmBound)
    end

    info.time_x_update += time_gen.time + time_br.time
    info.user.time_generators += time_gen.time
    info.user.time_branches += time_br.time
    return
end