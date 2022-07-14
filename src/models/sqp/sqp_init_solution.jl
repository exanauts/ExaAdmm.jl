"""
    init_solution_sqp()

- initialize sol for sqp
- initialize param for sqp
"""

function init_solution_sqp!(
    model::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::SolutionOneLevel{Float64,Array{Float64,1}}
    )

    ngen = model.grid_data.ngen
    nline = model.grid_data.nline
    nbus = model.grid_data.nbus

    brBusIdx = model.grid_data.brBusIdx
    Vmax = model.grid_data.Vmax; Vmin = model.grid_data.Vmin
    YffR = model.grid_data.YffR; YffI = model.grid_data.YffI
    YttR = model.grid_data.YttR; YttI = model.grid_data.YttI
    YftR = model.grid_data.YftR; YftI = model.grid_data.YftI
    YtfR = model.grid_data.YtfR; YtfI = model.grid_data.YtfI

    for g=1:ngen
    model.pg_sol[g] = 0.5*(model.grid_data.pgmin[g] + model.grid_data.pgmax[g])
    model.qg_sol[g] = 0.5*(model.grid_data.qgmin[g] + model.grid_data.qgmax[g])
    end

    for l=1:nline
        
        wij0 = (Vmax[brBusIdx[2*(l-1)+1]]^2 + Vmin[brBusIdx[2*(l-1)+1]]^2) / 2
        wji0 = (Vmax[brBusIdx[2*l]]^2 + Vmin[brBusIdx[2*l]]^2) / 2
        wR0 = sqrt(wij0 * wji0)

        model.line_fl[1, l] = YffR[l] * wij0 + YftR[l] * wR0  #p_ij 
        model.line_fl[2, l] = -YffI[l] * wij0 - YftI[l] * wR0 #q_ij
        model.line_fl[3, l] = YttR[l] * wji0 + YtfR[l] * wR0 #p_ji
        model.line_fl[4, l] = -YttI[l] * wji0 - YtfI[l] * wR0 #q_ji
      

        model.line_var[1, l] = wR0 #wRij
        model.line_var[2, l] = wR0 #wIij
        model.line_var[3, l] = wij0 #wi
        model.line_var[4, l] = wji0 #wj
        model.line_var[5, l] = 0.0 #theta_i
        model.line_var[6, l] = 0.0 #theta_j
    end

    for b=1:nbus
        model.w_sol[b] =( Vmax[b]^2 + Vmin[b]^2 )/2 
        model.theta_sol[b] = 0.0
    end
    #reset bool and multi and (qpsub) sol
    fill!(model.multi_line, 0.0)
    fill!(model.bool_line, false)
    fill!(sol, 0.0)

    return
end