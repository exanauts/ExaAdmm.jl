function admm_update_xbar(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelAcopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info = mod.solution, mod.info
    nblk_bus = div(mod.nbus, 32, RoundUp)

    time_bus = CUDA.@timed @cuda threads=32 blocks=nblk_bus bus_kernel_two_level_alternative(mod.baseMVA, mod.nbus, mod.gen_start, mod.line_start,
                                                                        mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                                                        mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr,
                                                                        sol.z_curr, sol.l_curr, sol.rho, mod.YshR, mod.YshI)
    info.time_xbar_update += time_bus.time
    info.user.time_buses += time_bus.time
    return
end