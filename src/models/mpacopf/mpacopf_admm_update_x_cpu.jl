# ------------------------------------------------------------------------
# Implementation of acopf_admm_update_x for MultiPeriodModel
# ------------------------------------------------------------------------

function mpacopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, info = env.params, mod.info

    acopf_admm_update_x_gen(env, mod.models[1], mod.models[1].gen_solution)
    for i=2:mod.len_horizon
        submod, subsol, sol_ramp, subdata = mod.models[i], mod.models[i].solution, mod.solution[i], mod.models[i].grid_data
        time_gen = @timed auglag_generator_kernel(3, subdata.ngen, submod.gen_start,
            info.inner, par.max_auglag, par.mu_max, 1.0,
            subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho,
            sol_ramp.u_curr, mod.models[i-1].solution.v_curr, sol_ramp.z_curr,
            sol_ramp.l_curr, sol_ramp.rho, sol_ramp.s_curr,
            submod.gen_membuf,
            subdata.pgmin, subdata.pgmax,
            subdata.qgmin, subdata.qgmax,
            subdata.ramp_rate,
            subdata.c2, subdata.c1, subdata.c0, subdata.baseMVA)

        submod.info.time_x_update += time_gen.time
        submod.info.user.time_generators += time_gen.time
    end
    return
end

function admm_update_x(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  device::Nothing=nothing
)
    mpacopf_admm_update_x_gen(env, mod)

    # Since temporal constraints are applied to generators only, line problems
    # can be solved by just reusing existing functions for a single period ACOPF.
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
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
    u, v, z, l, rho = mod.solution.u_curr, mod.solution.v_curr, mod.solution.z_curr, mod.solution.l_curr, mod.solution.rho
    r_v, r_z, r_l, r_rho = sol_ramp.v_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho
    c2, c1, baseMVA = mod.grid_data.c2, mod.grid_data.c1, mod.grid_data.baseMVA

    for I=1:mod.grid_data.ngen
        pg_idx = mod.gen_start + 2*(I-1)
        qg_idx = mod.gen_start + 2*(I-1) + 1

        u[pg_idx] = max(mod.grid_data.pgmin[I],
                        min(mod.grid_data.pgmax[I],
                        (-(c1[I]*baseMVA + (l[pg_idx] + rho[pg_idx]*(-v[pg_idx] + z[pg_idx])) +
                            (r_l[I] + r_rho[I]*(-r_v[I] + r_z[I])))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx] + r_rho[I])))
        u[qg_idx] = max(mod.grid_data.qgmin[I],
                        min(mod.grid_data.qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-v[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end
end

function acopf_admm_update_x_gen_between(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
end

function acopf_admm_update_x_gen_last(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol_ramp::SolutionRamping{Float64,Array{Float64,1}}
)
end

function acopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
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

function admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelMpacopfLoose{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    for i=1:mod.len_horizon
        admm_two_level(env, mod.models[i])
    end
    return
end