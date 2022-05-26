function admm_update_xbar(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::AbstractOPFModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info, data = mod.solution, mod.info, mod.grid_data
    nblk_bus = div(data.nbus, 32, RoundUp)

    time_bus = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_two_level_alternative(data.baseMVA, data.nbus, mod.gen_start, mod.line_start,
                                                                        data.FrStart, data.FrIdx, data.ToStart, data.ToIdx, data.GenStart,
                                                                        data.GenIdx, data.Pd, data.Qd, sol.u_curr, sol.v_curr,
                                                                        sol.z_curr, sol.l_curr, sol.rho, data.YshR, data.YshI)
    info.time_xbar_update += time_bus.time
    info.user.time_buses += time_bus.time
    return
end
