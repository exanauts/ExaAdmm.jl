function init_ramp_solution_kernel(
    n::Int, gen_start::Int,
    u_ramp::CuDeviceArray{Float64,1}, s_ramp::CuDeviceArray{Float64,1},
    u_curr::CuDeviceArray{Float64,1}, v_prev::CuDeviceArray{Float64,1}
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= n
        gid = gen_start + 2*(tx-1)
        u_ramp[tx] = v_prev[gid]
        s_ramp[tx] = u_curr[gid] - v_prev[gid]
    end
    return
end

function init_solution!(
    mod::MultiPeriodModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol_ramp::Vector{SolutionRamping{Float64,CuArray{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        fill!(sol_ramp[i], 0.0)
        sol_ramp[i].rho .= rho_pq
    end

    # Set initial point for ramp variables.
    for i=2:mod.len_horizon
        CUDA.@sync @cuda threads=64 blocks=(div(mod.models[i].ngen-1,64)+1) init_ramp_solution_kernel(
            mod.models[i].ngen, mod.models[i].gen_start,
            sol_ramp[i].u_curr, sol_ramp[i].s_curr,
            mod.models[i].solution.u_curr, mod.models[i-1].solution.v_curr
        )
    end
    return
end