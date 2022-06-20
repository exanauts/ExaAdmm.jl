function admm_update_xbar(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device::KA.GPU
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    # nblk_bus = div(data.nbus, 32, RoundUp)

    wait(bus_kernel_two_level_alternative_ka(device,32,data.nbus)(
            data.baseMVA, data.nbus, mod.gen_start, mod.line_start,
            data.FrStart, data.FrIdx, data.ToStart, data.ToIdx, data.GenStart,
            data.GenIdx, data.Pd, data.Qd, sol.u_curr, sol.v_curr,
            sol.z_curr, sol.l_curr, sol.rho, data.YshR, data.YshI,
            dependencies=Event(device)
        )
    )
    info.time_xbar_update += 0.0
    info.user.time_buses += 0.0
    return
end
