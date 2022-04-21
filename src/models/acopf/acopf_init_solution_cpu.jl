function init_solution!(
    model::ModelAcopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::SolutionOneLevel{Float64,Array{Float64,1}},
    rho_pq::Float64, rho_va::Float64
)

    ngen = model.ngen
    nline = model.nline

    brBusIdx = model.brBusIdx
    Vmax = model.Vmax; Vmin = model.Vmin
    YffR = model.YffR; YffI = model.YffI
    YttR = model.YttR; YttI = model.YttI
    YftR = model.YftR; YftI = model.YftI
    YtfR = model.YtfR; YtfI = model.YtfI

    fill!(sol, 0.0)
    sol.rho .= rho_pq

    for g=1:ngen
        pg_idx = model.gen_start + 2*(g-1)
        sol.v_curr[pg_idx] = 0.5*(model.pgmin[g] + model.pgmax[g])
        sol.v_curr[pg_idx+1] = 0.5*(model.qgmin[g] + model.qgmax[g])
    end

    for l=1:nline
        wij0 = (Vmax[brBusIdx[2*(l-1)+1]]^2 + Vmin[brBusIdx[2*(l-1)+1]]^2) / 2
        wji0 = (Vmax[brBusIdx[2*l]]^2 + Vmin[brBusIdx[2*l]]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        pij_idx = model.line_start + 8*(l-1)
        sol.v_curr[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0
        sol.v_curr[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0
        sol.v_curr[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0
        sol.v_curr[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0
        sol.v_curr[pij_idx+4] = wij0
        sol.v_curr[pij_idx+5] = wji0
        sol.v_curr[pij_idx+6] = 0.0
        sol.v_curr[pij_idx+7] = 0.0

        sol.rho[pij_idx+4:pij_idx+7] .= rho_va
    end

    return
end