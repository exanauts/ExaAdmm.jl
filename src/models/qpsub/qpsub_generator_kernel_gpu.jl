"""
    generator_kernel_two_level_qpsub()

record cpu time: return tcpu
"""

function generator_kernel_two_level(
    model::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
    )
    nblk = div(model.grid_data.ngen, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk generator_kernel_two_level(baseMVA, model.grid_data.ngen, model.gen_start,
    u, xbar, zu, lu, rho_u, model.qpsub_pgmin, model.qpsub_pgmax, model.qpsub_qgmin, model.qpsub_qgmax,
    model.qpsub_c2, model.qpsub_c1)
    return tgpu
end
