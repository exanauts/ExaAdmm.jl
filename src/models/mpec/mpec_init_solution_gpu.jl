function mpec_init_solution_generator(
    ngen::Int, gen_start::Int, rho_va::Float64, rho_pq::Float64,
    v::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    vgmin::CuDeviceArray{Float64,1}, vgmax::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= ngen
        pg_idx = gen_start + 2*(I-1)
        vg_idx = gen_start + 2*ngen + (I-1)
        fg_idx = gen_start + 3*ngen + (I-1)
        v[pg_idx] = 0.5*(pgmin[I] + pgmax[I])
        v[pg_idx+1] = 0.5*(qgmin[I] + qgmax[I])
        v[vg_idx] = (0.5*(vgmin[I] + vgmax[I]))^2
        v[fg_idx] = 0.0
        rho[vg_idx] = rho_va*1e1
        rho[fg_idx] = rho_pq*1e1
    end
    return
end

function mpec_init_solution_line(
    nline::Int, line_start::Int, rho_va::Float64,
    v::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    brBusIdx::CuDeviceArray{Int,1},
    Vmin::CuDeviceArray{Float64,1}, Vmax::CuDeviceArray{Float64,1},
    YffR::CuDeviceArray{Float64,1}, YffI::CuDeviceArray{Float64,1},
    YftR::CuDeviceArray{Float64,1}, YftI::CuDeviceArray{Float64,1},
    YtfR::CuDeviceArray{Float64,1}, YtfI::CuDeviceArray{Float64,1},
    YttR::CuDeviceArray{Float64,1}, YttI::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= nline
        wij0 = (Vmax[brBusIdx[2*(I-1)+1]]^2 + Vmin[brBusIdx[2*(I-1)+1]]^2) / 2
        wji0 = (Vmax[brBusIdx[2*I]]^2 + Vmin[brBusIdx[2*I]]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = line_start + 8*(I-1)
        v[pij_idx] = YffR[I] * wij0 + YftR[I] * wR0
        v[pij_idx+1] = -YffI[I] * wij0 - YftI[I] * wR0
        v[pij_idx+2] = YttR[I] * wji0 + YtfR[I] * wR0
        v[pij_idx+3] = -YttI[I] * wji0 - YtfI[I] * wR0
        v[pij_idx+4] = wij0
        v[pij_idx+5] = wji0
        v[pij_idx+6] = 0.0
        v[pij_idx+7] = 0.0

        rho[pij_idx+4:pij_idx+7] .= rho_va
    end
    return
end

function init_solution!(
    model::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol::Solution{Float64,CuArray{Float64,1}},
    rho_pq::Float64, rho_va::Float64
)
    grid = model.grid

    fill!(sol, 0.0)
    sol.rho .= rho_pq

    nblk_gen = div(grid.ngen-1, 32)+1
    nblk_line = div(grid.nline-1, 32)+1

    @cuda threads=32 blocks=nblk_gen mpec_init_solution_generator(
        grid.ngen, model.gen_start, rho_va, rho_pq,
        sol.v_curr, sol.rho,
        grid.pgmin, grid.pgmax,
        grid.qgmin, grid.qgmax,
        grid.vgmin, grid.vgmax
    )

    @cuda threads=32 blocks=nblk_line mpec_init_solution_line(
        grid.nline, model.line_start, rho_va,
        sol.v_curr, sol.rho,
        grid.brBusIdx,
        grid.Vmin, grid.Vmax,
        grid.YffR, grid.YffI,
        grid.YftR, grid.YftI,
        grid.YtfR, grid.YtfI,
        grid.YttR, grid.YttI
    )
    CUDA.synchronize()

    return
end
