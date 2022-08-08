function uc_init_mpmodel_solution!(
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionRamping{Float64,Array{Float64,1}}},
    uc_sol::Vector{SolutionUC{Float64,Array{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        init_solution!(mod.models[i], mod.models[i].solution, rho_pq, rho_va)

        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
        griddata = mod.models[1].grid_data
        if i > 1
            uc_sol_i = uc_sol[i]
            for g=1:griddata.ngen
                Ri = griddata.ramp_rate[g] # Assume RU = RD = Ri
                Si = Ri # Assume SU = SD = Si. TODO: account for startup/shutdown ramping later. For now assume Si = Ri.
                gen_start = mod.models[i].gen_start
                p_prev = mod.models[i-1].solution.v_curr[gen_start+2*(g-1)]
                q_prev = mod.models[i-1].solution.v_curr[gen_start+2*(g-1)+1]
                p_curr = mod.models[i].solution.u_curr[gen_start+2*(g-1)]
                q_curr = mod.models[i].solution.u_curr[gen_start+2*(g-1)+1]
                v, w, y = uc_sol_i.v_curr[3*g-2], uc_sol_i.v_curr[3*g-1], uc_sol_i.v_curr[3*g]
                vhat = uc_sol[i-1].v_curr[3*g-2]
                Pmax, Pmin, Qmax, Qmin = griddata.pgmax[g], griddata.pgmin[g], griddata.qgmax[g], griddata.qgmin[g]
                # u_curr: [p, q]
                sol[i].u_curr[2*g-1] = p_prev
                sol[i].u_curr[2*g] = q_prev
                # s_curr: [s^U, s^D, s^{p,UB}, s^{p,LB}, s^{q,UB}, s^{q,LB}]
                sol[i].s_curr[6*g-5] = p_curr - p_prev - Ri*vhat - Si*w
                sol[i].s_curr[6*g-4] = -(p_curr - p_prev + Ri*v + Si*y)
                sol[i].s_curr[6*g-3] = p_curr - Pmax*v 
                sol[i].s_curr[6*g-2] = -(p_curr - Pmin*v)
                sol[i].s_curr[6*g-1] = q_curr - Qmax*v
                sol[i].s_curr[6*g] = -(q_curr - Qmin*v)
            end
        end
    end
end


function init_solution!(
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionUC{Float64,Array{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    uc_solution = mod.uc_solution
    uc_init_mpmodel_solution!(mod.mpmodel, mod.mpmodel.solution, uc_solution, rho_pq, rho_va)
    # TODO: think about how to initialize UC solution
    # One idea: initialize with the status of previous time step?
    for i=1:mod.mpmodel.len_horizon
        sol[i].rho .= rho_pq
    end
end