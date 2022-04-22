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
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol::Vector{SolutionRamping{Float64,CuArray{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        init_solution!(mod.models[i], mod.models[i].solution, rho_pq, rho_va)

        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
        if i > 1
            CUDA.@sync @cuda threads=64 blocks=(div(mod.models[i].ngen-1,64)+1) init_ramp_solution_kernel(
                mod.models[i].ngen, mod.models[i].gen_start,
                sol[i].u_curr, sol[i].s_curr,
                mod.models[i].solution.u_curr, mod.models[i-1].solution.v_curr
            )
        end
    end

    return
end
