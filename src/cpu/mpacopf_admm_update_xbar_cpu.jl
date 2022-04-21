function acopf_admm_update_xbar(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon-1
        submod, subsol, sol_ramp = mod.models[i], mod.models[i].solution, mod.solution[i+1]
        bus_time = @timed bus_kernel_ramp(submod.baseMVA, submod.nbus, submod.gen_start, submod.line_start,
            submod.FrStart, submod.FrIdx, submod.ToStart, submod.ToIdx, submod.GenStart,
            submod.GenIdx, submod.Pd, submod.Qd, subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho, submod.YshR, submod.YshI,
            sol_ramp.u_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho)

        submod.info.time_xbar_update += bus_time.time
        submod.info.user.time_buses += bus_time.time
    end

    # The last one is not related to ramping, therefore, we reuse the existing implementation.
    acopf_admm_update_xbar(env, mod.models[mod.len_horizon])
    return
end