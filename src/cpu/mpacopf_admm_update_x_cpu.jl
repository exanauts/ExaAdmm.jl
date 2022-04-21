# ------------------------------------------------------------------------
# Implementation of acopf_admm_update_x for MultiPeriodModel
# ------------------------------------------------------------------------

function mpacopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, info = env.params, mod.info

    acopf_admm_update_x_gen(env, mod.models[1], mod.models[1].gen_solution)
    for i=2:mod.len_horizon
        submod, subsol, sol_ramp = mod.models[i], mod.models[i].solution, mod.solution[i]
        time_gen = @timed auglag_generator_kernel(3, submod.ngen, submod.gen_start,
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

        submod.info.time_x_update += time_gen.time
        submod.info.user.time_generators += time_gen.time
    end
    return
end

function acopf_admm_update_x(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    mpacopf_admm_update_x_gen(env, mod)
    for i=1:mod.len_horizon
        acopf_admm_update_x_line(env, mod.models[i])
    end
    return
end

# ------------------------------------------------------------------------
# Implementation of acopf_admm_update_x for MultiPeriodModelLoose
# ------------------------------------------------------------------------

function acopf_admm_update_x_gen_first(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
    u, v, z, l, rho = mod.solution.u_curr, mod.solution.v_curr, mod.solution.z_curr, mod.solution.l_curr, mod.solution.rho
    r_v, r_z, r_l, r_rho = sol_ramp.v_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho
    c2, c1, baseMVA = mod.c2, mod.c1, mod.baseMVA

    for I=1:mod.ngen
        pg_idx = mod.gen_start + 2*(I-1)
        qg_idx = mod.gen_start + 2*(I-1) + 1

        u[pg_idx] = max(mod.pgmin[I],
                        min(mod.pgmax[I],
                        (-(c1[I]*baseMVA + (l[pg_idx] + rho[pg_idx]*(-v[pg_idx] + z[pg_idx])) +
                            (r_l[I] + r_rho[I]*(-r_v[I] + r_z[I])))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx] + r_rho[I])))
        u[qg_idx] = max(mod.qgmin[I],
                        min(mod.qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-v[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end
end

function acopf_admm_update_x_gen_between(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
end

function acopf_admm_update_x_gen_last(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
end

function acopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
    if sol_ramp.t == 1
        acopf_admm_update_x_gen_first(env, mod, sol_ramp)
    elseif sol_ramp.t < sol_ramp.len_horizon
        acopf_admm_update_x_gen_between(env, mod, sol_ramp)
    else
        acopf_admm_update_x_gen_last(env, mod, sol_ramp)
    end
    return
end

function acopf_admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::MultiPeriodModelLoose{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon
        admm_restart(env, mod.models[i])
    end
    return
end