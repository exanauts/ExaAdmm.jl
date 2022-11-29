# TODO: initialization for uc_sol.u_curr seems meaningless here... will skip for now

function uc_init_mpmodel_solution!(
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionRamping{Float64,Array{Float64,1}}},
    vr_sol::Vector{SolutionRamping{Float64,Array{Float64,1}}},
    uc_sol::SolutionUC{Float64,Array{Float64,2}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        init_solution!(mod.models[i], mod.models[i].solution, rho_pq, rho_va)
        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
        griddata = mod.models[1].grid_data

        for g=1:griddata.ngen
            Ri = griddata.ramp_rate[g] # Assume RU = RD = Ri
            Si = Ri # Assume SU = SD = Si. TODO: account for startup/shutdown ramping later. For now assume Si = Ri.
            gen_start = mod.models[i].gen_start
            p_curr = mod.models[i].solution.u_curr[gen_start+2*(g-1)]
            q_curr = mod.models[i].solution.u_curr[gen_start+2*(g-1)+1]
            v, w, y = uc_sol.v_curr[g, 3*i-2], uc_sol.v_curr[g, 3*i-1], uc_sol.v_curr[g, 3*i]
            if i > 1
                p_prev = mod.models[i-1].solution.v_curr[gen_start+2*(g-1)]
                vhat = uc_sol.v_curr[g, 3*(i-1)-2]
                gen_start = mod.models[i].gen_start
                sol[i].u_curr[g] = mod.models[i-1].solution.v_curr[gen_start+2*(g-1)]
                vr_sol[i].u_curr[g] = vhat
                sol[i].s_curr[2*g-1] = p_curr - p_prev - Ri*vhat - Si*w
                sol[i].s_curr[2*g] = -(p_curr - p_prev + Ri*v + Si*y)    
            end
            Pmax, Pmin, Qmax, Qmin = griddata.pgmax[g], griddata.pgmin[g], griddata.qgmax[g], griddata.qgmin[g]
            uc_sol.u_curr[g, 3*i-2] = v
            uc_sol.u_curr[g, 3*i-1] = w
            uc_sol.u_curr[g, 3*i] = y
            # s_curr: [s^{p,UB}, s^{p,LB}, s^{q,UB}, s^{q,LB}]
            uc_sol.s_curr[g, 4*i-3] = p_curr - Pmax*v 
            uc_sol.s_curr[g, 4*i-2] = -(p_curr - Pmin*v)
            uc_sol.s_curr[g, 4*i-1] = q_curr - Qmax*v
            uc_sol.s_curr[g, 4*i] = -(q_curr - Qmin*v)
        end
    end
end


function init_solution!(
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::SolutionUC{Float64,Array{Float64,2}},
    rho_pq::Float64, rho_va::Float64
)
    uc_solution = mod.uc_solution
    vr_solution = mod.vr_solution
    uc_init_mpmodel_solution!(mod.mpmodel, mod.mpmodel.solution, vr_solution, uc_solution, rho_pq, rho_va)
    for i in 1:mod.mpmodel.len_horizon
        fill!(vr_solution[i], 0.0)
    end
    sol.rho .= rho_pq
end