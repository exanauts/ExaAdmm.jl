function init_generator_kernel_one_level(n::Int, gen_start::Int,
    pgmax::CuDeviceArray{Float64,1}, pgmin::CuDeviceArray{Float64,1},
    qgmax::CuDeviceArray{Float64,1}, qgmin::CuDeviceArray{Float64,1},
    v::CuDeviceArray{Float64,1}
)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= n
        v[gen_start + 2*(g-1)] = 0.5*(pgmin[g] + pgmax[g])
        v[gen_start + 2*(g-1)+1] = 0.5*(qgmin[g] + qgmax[g])
    end

    return
end

function init_branch_bus_kernel_one_level(n::Int, line_start::Int, rho_va::Float64,
    brBusIdx::CuDeviceArray{Int,1},
    Vmax::CuDeviceArray{Float64,1}, Vmin::CuDeviceArray{Float64,1},
    YffR::CuDeviceArray{Float64,1}, YffI::CuDeviceArray{Float64,1},
    YftR::CuDeviceArray{Float64,1}, YftI::CuDeviceArray{Float64,1},
    YtfR::CuDeviceArray{Float64,1}, YtfI::CuDeviceArray{Float64,1},
    YttR::CuDeviceArray{Float64,1}, YttI::CuDeviceArray{Float64,1},
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1}
)
    l = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if l <= n
        wij0 = (Vmax[brBusIdx[2*(l-1)+1]]^2 + Vmin[brBusIdx[2*(l-1)+1]]^2) / 2
        wji0 = (Vmax[brBusIdx[2*l]]^2 + Vmin[brBusIdx[2*l]]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = line_start + 8*(l-1)
        v[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        v[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        v[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        v[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0
        v[pij_idx+4] = wij0
        v[pij_idx+5] = wji0
        v[pij_idx+6] = 0.0
        v[pij_idx+7] = 0.0

        rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    return
end

function init_solution!(
    model::ModelAcopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol::SolutionOneLevel{Float64,CuArray{Float64,1}},
    rho_pq::Float64, rho_va::Float64
)
    fill!(sol, 0.0)
    sol.rho .= rho_pq

    @cuda threads=64 blocks=(div(model.ngen-1,64)+1) init_generator_kernel_one_level(model.ngen, model.gen_start,
                    model.pgmax, model.pgmin, model.qgmax, model.qgmin, sol.v_curr)
    @cuda threads=64 blocks=(div(model.nline-1,64)+1) init_branch_bus_kernel_one_level(model.nline, model.line_start, rho_va,
                model.brBusIdx, model.Vmax, model.Vmin, model.YffR, model.YffI, model.YftR, model.YftI,
                model.YtfR, model.YtfI, model.YttR, model.YttI, sol.u_curr, sol.v_curr, sol.rho)
    CUDA.synchronize()

end