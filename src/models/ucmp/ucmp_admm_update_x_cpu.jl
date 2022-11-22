# ------------------------------------------------------------------------
# Implementation of acopf_admm_update_x for UCMPModel
# ------------------------------------------------------------------------

function ucmp_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, info = env.params, mod.info
    mpmod = mod.mpmodel
    fill!(mod.uc_membuf, 0)
    uc_sol = mod.uc_solution

    for i=1:mpmod.len_horizon
        submod, subsol, sol_ramp, subdata = mpmod.models[i], mpmod.models[i].solution, mpmod.solution[i], mpmod.models[i].grid_data
        v_ramp = mod.vr_solution[i]
        time_gen = @timed ucmp_auglag_generator_kernel(i, 13, subdata.ngen, submod.gen_start,
            info.inner, par.max_auglag, par.mu_max, 1.0,
            subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho,
            sol_ramp.u_curr, sol_ramp.v_curr, sol_ramp.z_curr,
            sol_ramp.l_curr, sol_ramp.rho, sol_ramp.s_curr,
            v_ramp.u_curr, v_ramp.v_curr, v_ramp.z_curr,
            v_ramp.l_curr, v_ramp.rho, v_ramp.s_curr,
            uc_sol.u_curr, uc_sol.v_curr, uc_sol.z_curr,
            uc_sol.l_curr, uc_sol.rho, uc_sol.s_curr,
            submod.gen_membuf,
            subdata.pgmin, subdata.pgmax,
            subdata.qgmin, subdata.qgmax,
            subdata.ramp_rate,
            subdata.c2, subdata.c1, subdata.c0, subdata.baseMVA)

        submod.info.time_x_update += time_gen.time
        submod.info.user.time_generators += time_gen.time

        if i > 1
            ucmp_update_uc_membuf_with_ramping_kernel(
                i, subdata.ngen,
                sol_ramp.u_curr, sol_ramp.z_curr,
                sol_ramp.l_curr, sol_ramp.rho,
                mod.uc_membuf
            )
        end
    end
    return
end

function admm_update_x(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  device::Nothing=nothing
)
    ucmp_admm_update_x_gen(env, mod)

    # Since temporal constraints are applied to generators only, line problems
    # can be solved by just reusing existing functions for a single period ACOPF.
    for i=1:mod.mpmodel.len_horizon
        acopf_admm_update_x_line(env, mod.mpmodel.models[i])
    end
    return
end