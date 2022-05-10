## current main function (similar to solve_acopf) 
# test each QP subproblem functions here 
function solve_qpsub(case::String; case_format="matpower", verbose = 1
)
    # test each QP subproblem functions here 
    T = Float64; TD = Array{Float64,1}; TI = Array{Int64,1}; TM = Array{Float64,2}; TS = Array{Float64,3}

    rho_pq=400.0; rho_va=40000.0 #Place Holder  

    # env = AdmmEnvSQP{T,TD,TI,TM}(case; case_format=case_format, verbose = verbose) #new environ 
    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va) #old environ
    mod_old = ModelAcopf{T,TD,TI,TM}(env) #old model help debug and understand 
    
    mod = ModelQpsub{T,TD,TI,TM,TS}(env)

    env2 = SolutionQP_gen{T,TD}(mod.ngen)

    env3 = SolutionACOPF{T,TD}(mod.ngen)

    env4 = Lam_rho_pi_gen{T,TD}(mod.ngen)
     
    env5 = Coeff_SQP{T,TD,TM,TS}(mod.ngen,mod.nline,mod.nbus) 

    env6 = SolutionQP_bus{T,TD}(mod.ngen, mod.nbus)

    env7 = SolutionQP_br{T,TD}(mod.nline)

    #generator 
    println(mod.gen_qp)

    tgen = generatorQP(env,mod)

    println(mod.gen_qp)

    #branch

    #bus problem 
    
    return env, env2, env3, env4, env5, env6, env7, mod, tgen, mod_old 
end

## one_line_test 
# env, env2, env3, env4, env5, env6, env7, mod, tgen, mod_old = ExaAdmm.solve_qpsub("case9.m");
# ExaAdmm.generatorQP(env,mod)

# env, mod = ExaAdmm.solve_acopf("case1354pegase.m");