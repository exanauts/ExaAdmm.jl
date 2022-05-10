function init_solution!(
    mod::ModelMpacopf{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionRamping{Float64,Array{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        init_solution!(mod.models[i], mod.models[i].solution, rho_pq, rho_va)

        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
        if i > 1
            for g=1:mod.models[i].grid_data.ngen
                gen_start = mod.models[i].gen_start
                sol[i].u_curr[g] = mod.models[i-1].solution.v_curr[gen_start+2*(g-1)]
                sol[i].s_curr[g] = mod.models[i].solution.u_curr[gen_start+2*(g-1)] - sol[i].u_curr[g]
            end
        end
    end
end

function init_solution!(
    mod::ModelMpacopfLoose{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionRamping{Float64,Array{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
    end

    # Set initial point for ramp variables.
    for i=2:mod.len_horizon
        for g=1:mod.models[i].grid_data.ngen
            gen_start = mod.models[i].gen_start
            sol[i].u_curr[g] = mod.models[i-1].solution.u_curr[gen_start+2*(g-1)]
            sol[i].v_curr[g] = mod.models[i-1].solution.u_curr[gen_start+2*(g-1)]
            sol[i].s_curr[g] = mod.models[i].solution.u_curr[gen_start+2*(g-1)] - sol[i].u_curr[g]
        end
    end
end