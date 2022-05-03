function mpacopf_admm_update_x_gen(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, info = env.params, mod.info

    shmem_size = env.params.gen_shmem_size
    ngen = mod.models[1].ngen
    acopf_admm_update_x_gen(env, mod.models[1], mod.models[1].gen_solution)

    for i=2:mod.len_horizon
        submod, subsol, sol_ramp = mod.models[i], mod.models[i].solution, mod.solution[i]
        time_gen = @timed begin

        @cuda threads=32 blocks=ngen shmem=shmem_size auglag_generator_kernel(
            3, submod.ngen, submod.gen_start,
            info.inner, par.max_auglag, par.mu_max, 1.0,
            subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho,
            sol_ramp.u_curr, mod.models[i-1].solution.v_curr, sol_ramp.z_curr,
            sol_ramp.l_curr, sol_ramp.rho, sol_ramp.s_curr,
            submod.gen_membuf,
            submod.pgmin, submod.pgmax,
            submod.qgmin, submod.qgmax,
            submod.ramp_rate,
            submod.c2, submod.c1, submod.c0, submod.baseMVA)
        CUDA.synchronize()
        end

        submod.info.time_x_update += time_gen.time
        submod.info.user.time_generators += time_gen.time
    end
    return
end

function admm_update_x(
  env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
  mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    mpacopf_admm_update_x_gen(env, mod)
    for i=1:mod.len_horizon
        acopf_admm_update_x_line(env, mod.models[i])
    end
    return
end