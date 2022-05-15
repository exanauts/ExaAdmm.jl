"""
    generator_kernel_two_level_qpsub()

update sol.u[pg_idx] and sol.u[qp_idx]
"""

function generator_kernel_two_level_qpsub(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z, l, rho,
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1}
)
    for I=1:ngen
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
    model::ModelQpsub{Float64,Array{Float64,1},Array{Int,1}},
    baseMVA::Float64, u, xbar, zu, lu, rho_u
)
tcpu = @timed generator_kernel_two_level_qpsub(baseMVA, model.grid_data.ngen, model.gen_start,
u, xbar, zu, lu, rho_u, model.pgmin_curr, model.pgmax_curr, model.grid_data.qgmin, model.grid_data.qgmax, 
model.grid_data.c2, model.grid_data.c1)
return tcpu
end