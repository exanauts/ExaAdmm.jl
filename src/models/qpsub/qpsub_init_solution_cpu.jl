"""
    init_solution()

- initialize sol.v_curr and sol.rho for all coupling
- initialize sqp_line, supY  
"""

function init_solution!(
    model::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Solution{Float64,Array{Float64,1}},
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
    
    fill!(sol, 0.0)
    fill!(model.lambda, 0.0)

    #qpsub var
    sol.rho .= rho_pq

    for g=1:ngen
        pg_idx = model.gen_start + 2*(g-1)
        sol.v_curr[pg_idx] = 0.5*(model.qpsub_pgmin[g] + model.qpsub_pgmax[g]) 
        sol.v_curr[pg_idx+1] = 0.5*(model.qpsub_qgmin[g] + model.qpsub_qgmax[g]) 
    end

    for l=1:nline

        pij_idx = model.line_start + 8*(l-1)

        supY = [YftR[l] YftI[l] YffR[l] 0 0 0;
               -YftI[l] YftR[l] -YffI[l] 0 0 0;
               YtfR[l] -YtfI[l] 0 YttR[l] 0 0;
               -YtfI[l] -YtfR[l] 0 -YttI[l] 0 0] #wijR, wijI, wi, wj, theta_i, theta_j

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