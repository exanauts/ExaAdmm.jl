function admm_update_xbar(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
  )
    for i=1:mod.len_horizon-1
        submod, subsol, sol_ramp, subdata = mod.models[i], mod.models[i].solution, mod.solution[i+1], mod.models[i].grid_data
        nblk_bus = div(mod.models[i].grid_data.nbus, 32, RoundUp)
        time_bus = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_ramp(
            subdata.baseMVA, subdata.nbus, submod.gen_start, submod.line_start,
            subdata.FrStart, subdata.FrIdx, subdata.ToStart, subdata.ToIdx, subdata.GenStart,
            subdata.GenIdx, subdata.Pd, subdata.Qd, subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho, subdata.YshR, subdata.YshI,
            sol_ramp.u_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho)

        submod.info.time_xbar_update += time_bus.time
        submod.info.user.time_buses += time_bus.time
    end

    # The last one is not related to ramping, therefore, we reuse the existing implementation.
    admm_update_xbar(env, mod.models[mod.len_horizon])
    return
  end