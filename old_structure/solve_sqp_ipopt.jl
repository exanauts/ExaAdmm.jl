"""
    solve_sqp_ipopt()
    
- full sqp with qpsub, soc, FR, merit, linfeas in ipopt  
- TODO: use the structure and convert to ExaAdmm
"""

function solve_sqp_ipopt(case::String;
    case_format="matpower", TR = 1.0, iter_lim = 100, eps = 1e-3
)
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}

    env = AdmmEnv{T,TD,TI,TM}(case, 400.0, 40000.0; case_format=case_format) #? 400.0 40000.0 place holder not used 
    mod = ModelQpsub{T,TD,TI,TM}(env; TR = TR, iter_lim = iter_lim, eps = eps)

    init_solution_sqp!(mod, mod.solution)

    err_prev = eval_linfeas_err(mod)

    lin_feas(mod)

    err_curr = eval_linfeas_err(mod)

    println("prev err = ",err_prev)
    println("curr err = ",err_curr)
    
    return env, mod
end