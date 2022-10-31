"""
Update variable `xbar`, representing the variables for buses in the component-based decomposition of ACOPF.
"""
function admm_update_xbar(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    for i=1:mod.mpmodel.len_horizon-1
        submod, subsol, sol_ramp, subdata = mod.mpmodel.models[i], mod.mpmodel.models[i].solution, mod.mpmodel.solution[i+1], mod.mpmodel.models[i].grid_data
        bus_time = @timed bus_kernel_ramp(subdata.baseMVA, subdata.nbus, submod.gen_start, submod.line_start,
            subdata.FrStart, subdata.FrIdx, subdata.ToStart, subdata.ToIdx, subdata.GenStart,
            subdata.GenIdx, subdata.Pd, subdata.Qd, subsol.u_curr, subsol.v_curr, subsol.z_curr,
            subsol.l_curr, subsol.rho, subdata.YshR, subdata.YshI,
            sol_ramp.u_curr, sol_ramp.z_curr, sol_ramp.l_curr, sol_ramp.rho,
            :ucmp)

        submod.info.time_xbar_update += bus_time.time
        submod.info.user.time_buses += bus_time.time
    end

    # The last one is not related to ramping, therefore, we reuse the existing implementation.
    # TODO: Double check this one?
    admm_update_xbar(env, mod.mpmodel.models[mod.mpmodel.len_horizon])

    # DP solution for the unit commitment part
    uc_params = mod.uc_params
    ngen = mod.mpmodel.models[1].grid_data.ngen
    uc_sol = mod.uc_solution
    dp_time = @timed dp_generator_kernel(
        ngen, mod.mpmodel.len_horizon,
        uc_params.v0, uc_params.Tu, uc_params.Td, uc_params.Hu, uc_params.Hd,
        uc_params.con, uc_params.coff,
        uc_sol.u_curr, uc_sol.v_curr, uc_sol.z_curr,
        uc_sol.l_curr, uc_sol.rho,
        mod.uc_membuf
    )

    return
end
