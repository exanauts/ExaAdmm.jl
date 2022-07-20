"""
    admm_update_xbar()

- update xbar: call bus_kernel_two_level_alternative_qpsub() = update sol.v (full) 
- record run time info.time_xbar_update, info.user.time_buses
"""



function admm_update_xbar(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
sol, info, data = mod.solution, mod.info, mod.grid_data
mod.v_prev .= sol.v_curr
bus_time = @timed bus_kernel_two_level_alternative_qpsub(data.baseMVA, data.nbus, mod.gen_start, mod.line_start,
                                    data.FrStart, data.FrIdx, data.ToStart, data.ToIdx, data.GenStart,
                                    data.GenIdx, mod.qpsub_Pd, mod.qpsub_Qd, sol.u_curr, sol.v_curr, sol.z_curr,
                                    sol.l_curr, sol.rho, data.YshR, data.YshI)
info.time_xbar_update += bus_time.time
    info.user.time_buses += bus_time.time
    return
end