function admm_update_xbar(
  env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
  mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    for i=1:mod.len_horizon-1
        submod, subsol, sol_ramp, subdata = mod.models[i], mod.models[i].solution, mod.solution[i+1], mod.models[i].grid_data
        bus_time = @timed bus_kernel_ramp(subdata.baseMVA, subdata.nbus, submod.gen_start, submod.line_start,
            subdata.FrStart, subdata.FrIdx, subdata.ToStart, subdata.ToIdx, subdata.GenStart,
            subdata.GenIdx, subdata.Pd, subdata.Qd, subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho, subdata.YshR, subdata.YshI,
            sol_ramp.u_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho)

        submod.info.time_xbar_update += bus_time.time
        submod.info.user.time_buses += bus_time.time
    end

    # The last one is not related to ramping, therefore, we reuse the existing implementation.
    admm_update_xbar(env, mod.models[mod.len_horizon])
    return
end