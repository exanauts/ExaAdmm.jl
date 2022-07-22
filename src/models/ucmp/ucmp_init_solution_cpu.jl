function init_solution!(
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    sol::Vector{SolutionUC{Float64,Array{Float64,1}}},
    rho_pq::Float64, rho_va::Float64
)
    init_solution!(mod.mpmodel, mod.mpmodel.solution, rho_pq, rho_va)
    # TODO: think about how to initialize UC solution
    # One idea: initialize with the status of previous time step?
    for i=1:mod.mpmodel.len_horizon
        sol[i].rho .= rho_pq
    end
end