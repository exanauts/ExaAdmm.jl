@kernel function generator_kernel_two_level_ka(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u, x, z,
    l, rho,
    pgmin, pgmax,
    qgmin, qgmax,
    c2, c1
)
    I_ = @index(Group, Linear)
    J_ = @index(Local, Linear)
    I = J_ + (@groupsize()[1] * (I_ - 1))
    if (I <= ngen)
        @inbounds begin
            pg_idx = gen_start + 2*(I-1)
            qg_idx = gen_start + 2*(I-1) + 1
            u[pg_idx] = max(pgmin[I],
                            min(pgmax[I],
                                (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))
            u[qg_idx] = max(qgmin[I],
                            min(qgmax[I],
                                (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
        end
    end
end

function generator_kernel_two_level(
    model::ModelAcopf,
    baseMVA::Float64, u, xbar,
    zu, lu, rho_u, device
)
    nblk = div(model.grid_data.ngen, 32, RoundUp)
    ev = generator_kernel_two_level_ka(device,32,nblk*32)(
        baseMVA, model.grid_data.ngen, model.gen_start,
        u, xbar, zu, lu, rho_u, model.pgmin_curr, model.pgmax_curr, model.grid_data.qgmin, model.grid_data.qgmax,
        model.grid_data.c2, model.grid_data.c1,
        dependencies=Event(device)
    )
    wait(ev)
    return 0.0
end
