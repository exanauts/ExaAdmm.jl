## current main function (similar to solve_acopf) 
# test each QP subproblem functions here 
function solve_qpsub(case::String; case_format="matpower", verbose = 1
)
    # test each QP subproblem functions here 
    T = Float64; TD = Array{Float64,1}; TI = Array{Int64,1}; TM = Array{Float64,2}
    env = AdmmEnvSQP{T,TD,TI,TM}(case; case_format=case_format, verbose = verbose)
    
    env2 = SolutionQP_gen{T,TD}(size(env.data.generators,1))

    env3 = SolutionACOPF{T,TD}(size(env.data.generators,1))

    env4 = Lam_rho_pi_gen{T,TD}(size(env.data.generators,1))
     
    env5 = Coeff_SQP{T,TD}(size(env.data.generators,1)) 

    env6 = SolutionQP_bus{T,TD}(size(env.data.generators,1))

    mod = ModelQpsub{T,TD,TI,TM}(env)

    #generator 
    println(mod.gen_qp)

    tgen = generatorQP(env,mod)

    println(mod.gen_qp)

    #branch

    #bus problem 
    
    return env, env2, env3, env4, env5, env6, mod, tgen 
end

## test code example 
# env, env2, env3, env4, env5, env6, mod, tgen = ExaAdmm.solve_qpsub("case9.m");
# ExaAdmm.generatorQP(env,mod)

# env, mod = ExaAdmm.solve_acopf("case1354pegase.m");