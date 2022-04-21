function acopf_admm_update_xbar(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol, info = mod.solution, mod.info
    bus_time = @timed bus_kernel_two_level_alternative(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                        mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                        mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr, sol.z_curr,
                                        sol.l_curr, sol.rho, mod.YshR, mod.YshI)
    info.time_xbar_update += bus_time.time
    info.user.time_buses += bus_time.time
    return
end