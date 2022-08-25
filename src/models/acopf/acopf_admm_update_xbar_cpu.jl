"""
Update variable `xbar`, representing the variables for buses in the component-based decomposition of ACOPF.
"""
function admm_update_xbar(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    bus_time = @timed bus_kernel_two_level_alternative(data.baseMVA, data.nbus, mod.gen_start, mod.line_start,
                                        data.FrStart, data.FrIdx, data.ToStart, data.ToIdx, data.GenStart,
                                        data.GenIdx, data.Pd, data.Qd, sol.u_curr, sol.v_curr, sol.z_curr,
                                        sol.l_curr, sol.rho, data.YshR, data.YshI)
    info.time_xbar_update += bus_time.time
    info.user.time_buses += bus_time.time
    return
end
