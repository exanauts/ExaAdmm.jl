"""
    init_solution()

- initialize sol.v_curr and sol.rho for all coupling
- Note: initialize sol.l, par.beta, sol.lz in SolutionOneLevel{T,TD}()
- initialize sqp variables as well  
"""

function init_solution!(
    model::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::SolutionOneLevel{Float64,Array{Float64,1}},
    rho_pq::Float64, rho_va::Float64
    )

    ngen = model.grid_data.ngen
    nline = model.grid_data.nline
    nbus = model.grid_data.nbus

    sqp_line = model.sqp_line
    ls = model.ls
    us = model.us

    brBusIdx = model.grid_data.brBusIdx
    Vmax = model.grid_data.Vmax; Vmin = model.grid_data.Vmin
    YffR = model.grid_data.YffR; YffI = model.grid_data.YffI
    YttR = model.grid_data.YttR; YttI = model.grid_data.YttI
    YftR = model.grid_data.YftR; YftI = model.grid_data.YftI
    YtfR = model.grid_data.YtfR; YtfI = model.grid_data.YtfI
    
    #write sqp var
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
    # fill!(model.multi_line, 1.0) #? tuning
    # fill!(model.bool_line, false)
    fill!(sol, 0.0)

    #qpsub var
    sol.rho .= rho_pq

    for g=1:ngen
        pg_idx = model.gen_start + 2*(g-1)
        sol.v_curr[pg_idx] = 0.5*(model.qpsub_pgmin[g] + model.qpsub_pgmax[g]) 
        sol.v_curr[pg_idx+1] = 0.5*(model.qpsub_qgmin[g] + model.qpsub_qgmax[g]) 
    end

    for l=1:nline
        #acopf
        # wij0 = (Vmax[brBusIdx[2*(l-1)+1]]^2 + Vmin[brBusIdx[2*(l-1)+1]]^2) / 2
        # wji0 = (Vmax[brBusIdx[2*l]]^2 + Vmin[brBusIdx[2*l]]^2) / 2
        # wR0 = sqrt(wij0 * wji0)

        pij_idx = model.line_start + 8*(l-1)
        # sol.v_curr[pij_idx] = YffR[l] * wij0 + YftR[l] * wR0  #p_ij 
        # sol.v_curr[pij_idx+1] = -YffI[l] * wij0 - YftI[l] * wR0 #q_ij
        # sol.v_curr[pij_idx+2] = YttR[l] * wji0 + YtfR[l] * wR0 #p_ji
        # sol.v_curr[pij_idx+3] = -YttI[l] * wji0 - YtfI[l] * wR0 #q_ji

        supY = [YftR[l] YftI[l] YffR[l] 0 0 0;
               -YftI[l] YftR[l] -YffI[l] 0 0 0;
               YtfR[l] -YtfI[l] 0 YttR[l] 0 0;
               -YtfI[l] -YtfR[l] 0 -YttI[l] 0 0]

        #initialize sqp_line 
        sqp_line[:,l] = (ls[l,:] + us[l,:])/2  # order |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|        

        sol.v_curr[pij_idx] = dot(supY[1,:],sqp_line[:,l])  #p_ij 
        sol.v_curr[pij_idx+1] = dot(supY[2,:],sqp_line[:,l]) #q_ij
        sol.v_curr[pij_idx+2] = dot(supY[3,:],sqp_line[:,l]) #p_ji
        sol.v_curr[pij_idx+3] = dot(supY[4,:],sqp_line[:,l]) #q_ji
        sol.v_curr[pij_idx+4] = sqp_line[3,l] #w_i
        sol.v_curr[pij_idx+5] = sqp_line[4,l] #w_j
        sol.v_curr[pij_idx+6] = sqp_line[5,l] #theta_i
        sol.v_curr[pij_idx+7] = sqp_line[6,l] #theta_j

        sol.rho[pij_idx:pij_idx+7] .= rho_va
    end

    return
end