@kernel function init_generator_kernel_one_level_ka(n::Int, gen_start::Int,
    pgmax, pgmin,
    qgmax, qgmin,
    v
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    g = J + (@groupsize()[1] * (I - 1))
    if g <= n
        @inbounds begin
            v[gen_start + 2*(g-1)] = 0.5*(pgmin[g] + pgmax[g])
            v[gen_start + 2*(g-1)+1] = 0.5*(qgmin[g] + qgmax[g])
        end
    end
end

@kernel function init_branch_bus_kernel_one_level_ka(n::Int, line_start::Int, rho_va::Float64,
    brBusIdx,
    Vmax, Vmin,
    YffR, YffI,
    YftR, YftI,
    YtfR, YtfI,
    YttR, YttI,
    u, v, rho
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    l = J + (@groupsize()[1] * (I - 1))
    if l <= n
        @inbounds begin
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

            rho[pij_idx+4] = rho_va
            rho[pij_idx+5] = rho_va
            rho[pij_idx+6] = rho_va
            rho[pij_idx+7] = rho_va
        end
    end
end

function init_solution!(
    model::AbstractOPFModel,
    sol::Solution,
    rho_pq::Float64, rho_va::Float64, device
)
    fill!(sol, 0.0)
    sol.rho .= rho_pq

    data = model.grid_data
    nblk_ngen = div(data.ngen-1,64)+1
    init_generator_kernel_one_level_ka(device, 64, 64*nblk_ngen)(
        data.ngen, model.gen_start,
        data.pgmax, data.pgmin, data.qgmax, data.qgmin, sol.v_curr
    )
    KA.synchronize(device)
    nblk_nline = div(data.nline-1,64)+1
    init_branch_bus_kernel_one_level_ka(device, 64, 64*nblk_nline)(
        data.nline, model.line_start, rho_va,
        data.brBusIdx, data.Vmax, data.Vmin, data.YffR, data.YffI, data.YftR, data.YftI,
        data.YtfR, data.YtfI, data.YttR, data.YttI, sol.u_curr, sol.v_curr, sol.rho
    )
    KA.synchronize(device)

end
