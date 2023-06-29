"""
    generator_kernel_two_level_qpsub()

record cpu time: return tcpu
"""

function generator_kernel_two_level(
    model::ModelQpsub,
    baseMVA::Float64, u, xbar, zu, lu, rho_u,
    device
    )
    nblk = div(model.grid_data.ngen, 32, RoundUp)
    ev = generator_kernel_two_level_ka(device, 32, nblk*32)(
        baseMVA, model.grid_data.ngen, model.gen_start,
        u, xbar, zu, lu, rho_u, model.qpsub_pgmin, model.qpsub_pgmax, model.qpsub_qgmin, model.qpsub_qgmax,
        model.qpsub_c2, model.qpsub_c1
    )
    KA.synchronize(device)
    return 0.0
end
