function mpec_frequency_update(
    ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1}
)
    freq_start = gen_start + 3*ngen
    rhs = 0.0
    rho_sum = 0.0
    for g=1:ngen
        rhs += l[freq_start+g-1] + rho[freq_start+g-1]*(u[freq_start+g-1] + z[freq_start+g-1])
        rho_sum += rho[freq_start+g-1]
    end
    freq = (rhs / rho_sum)
    v[freq_start:freq_start+ngen-1] .= freq
    return
end

function acopf_admm_update_xbar(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    grid, sol, info = mod.grid, mod.solution, mod.info
    nblk_bus = div(grid.nbus-1, 32)+1
    bus_time = CUDA.@timed @cuda threads=32 blocks=nblk_bus  bus_kernel_mpec(
        grid.baseMVA, grid.nbus, grid.ngen, mod.gen_start, mod.line_start,
        grid.FrStart, grid.FrIdx, grid.ToStart, grid.ToIdx, grid.GenStart,
        grid.GenIdx, grid.StoStart, grid.StoIdx, grid.Pd, grid.Qd,
        sol.u_curr, sol.v_curr, sol.z_curr,
        sol.l_curr, sol.rho, grid.YshR, grid.YshI
    )

    info.time_xbar_update += bus_time.time
    info.user.time_buses += bus_time.time

    freq_time = CUDA.@timed @cuda threads=32 blocks=1 mpec_frequency_update(
        grid.ngen, mod.gen_start,
        sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho
    )
    info.time_xbar_update += freq_time.time

    return
end