function init_solution!(
    model::ComplementarityModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Solution{Float64,Array{Float64,1}},
    rho_pq::Float64, rho_va::Float64
)

    grid = model.grid
    ngen = grid.ngen
    nline = grid.nline

    brBusIdx = grid.brBusIdx
    Vmax = grid.Vmax; Vmin = grid.Vmin
    YffR = grid.YffR; YffI = grid.YffI
    YttR = grid.YttR; YttI = grid.YttI
    YftR = grid.YftR; YftI = grid.YftI
    YtfR = grid.YtfR; YtfI = grid.YtfI

    fill!(sol, 0.0)
    sol.rho .= rho_pq

    for g=1:ngen
        pg_idx = model.gen_start + 2*(g-1)
        vg_idx = model.gen_start + 2*ngen + (g-1)
        fg_idx = model.gen_start + 3*ngen + (g-1)
        sol.v_curr[pg_idx] = 0.5*(grid.pgmin[g] + grid.pgmax[g])
        sol.v_curr[pg_idx+1] = 0.5*(grid.qgmin[g] + grid.qgmax[g])
        sol.v_curr[vg_idx] = (0.5*(grid.vgmin[g] + grid.vgmax[g]))^2
        sol.v_curr[fg_idx] = 0.0
        sol.rho[vg_idx] = rho_va*1e1
        sol.rho[fg_idx] = rho_pq*1e1
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
