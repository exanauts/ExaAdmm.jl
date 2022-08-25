"""
    generator_kernel_two_level_qpsub()

update sol.u[pg_idx] and sol.u[qp_idx]
"""

function generator_kernel_two_level_qpsub(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
    )
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I<=ngen
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[I],
                        min(pgmax[I],
                            (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end


"""
    generator_kernel_two_level_qpsub()

record cpu time: return tcpu 
"""

function generator_kernel_two_level_qpsub(
    model::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
    )
    nblk = div(model.grid_data.ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level_qpsub(baseMVA, model.grid_data.ngen, model.gen_start,
    u, xbar, zu, lu, rho_u, model.qpsub_pgmin, model.qpsub_pgmax, model.qpsub_qgmin, model.qpsub_qgmax, 
    model.qpsub_c2, model.qpsub_c1)
    return tgpu
end