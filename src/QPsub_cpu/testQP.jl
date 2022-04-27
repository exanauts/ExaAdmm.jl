## current main function (similar to solve_acopf) 
# test each QP subproblem functions here 
function testQP(case::String; case_format="matpower", verbose = 1
)
    # test each QP subproblem functions here 
    T = Float64; TD = Array{Float64,1}; TI = Array{Int64,1}; TM = Array{Float64,2}
    env = AdmmEnvSQP{T,TD,TI,TM}(case; case_format=case_format, verbose = verbose)
    
    env2 = SolutionQP_gen{T,TD}(size(env.data.generators,1))

    env3 = SolutionACOPF{T,TD}(size(env.data.generators,1))

    env4 = lam_rho_gen{T,TD}(size(env.data.generators,1))

    #generator 

    #branch

    #bus problem 
    
    return env, env2, env3, env4
end

