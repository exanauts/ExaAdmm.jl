"""
    generator_kernel_two_level_qpsub()

record cpu time: return tcpu
"""

function generator_kernel_two_level(
    model::ModelQpsub{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u,
    device::Nothing=nothing
)
tcpu = @timed generator_kernel_two_level(baseMVA, model.grid_data.ngen, model.gen_start,
u, xbar, zu, lu, rho_u, model.qpsub_pgmin, model.qpsub_pgmax, model.qpsub_qgmin, model.qpsub_qgmax,
model.qpsub_c2, model.qpsub_c1)
return tcpu
end
