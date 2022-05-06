function acopf_admm_update_xbar(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ComplementarityModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    grid, sol, info = mod.grid, mod.solution, mod.info
    bus_time = @timed bus_kernel_mpec(grid.baseMVA, grid.nbus, grid.ngen, mod.gen_start, mod.line_start,
                      grid.FrStart, grid.FrIdx, grid.ToStart, grid.ToIdx, grid.GenStart,
                      grid.GenIdx, grid.StoStart, grid.StoIdx, grid.Pd, grid.Qd,
                      sol.u_curr, sol.v_curr, sol.z_curr,
                      sol.l_curr, sol.rho, grid.YshR, grid.YshI)
    info.time_xbar_update += bus_time.time
    info.user.time_buses += bus_time.time

    freq_time = @timed begin
        ngen = grid.ngen
        rho_sum = 0.0
        rhs = 0.0

        u, v, l, z, rho = sol.u_curr, sol.v_curr, sol.l_curr, sol.z_curr, sol.rho

        freq_start = mod.gen_start + 3*ngen
        rhs = sum(l[freq_start+g-1] + rho[freq_start+g-1]*(u[freq_start+g-1] + z[freq_start+g-1]) for g=1:ngen)
        rho_sum = sum(rho[freq_start+g-1] for g=1:ngen)
        freq = (rhs / rho_sum)
        v[freq_start:freq_start+ngen-1] .= freq
    end
    info.time_xbar_update += freq_time.time

    return
end