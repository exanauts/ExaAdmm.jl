using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt


INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "matpower")

case = joinpath(INSTANCES_DIR, "case9.m")
# case = joinpath(INSTANCES_DIR, "case30.m")
# case = joinpath(INSTANCES_DIR, "case118.m"); fix_line = true
# case = joinpath(INSTANCES_DIR, "case300.m")
# case = joinpath(INSTANCES_DIR, "case1354pegase.m")
# case = joinpath(INSTANCES_DIR, "case2869pegase.m")





# Initialize an qpsub model with default options as shell for qpsub.
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}

    env1 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, 400.0, 40000.0) #? 400, 40000 place holder not used 
    mod1 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env1; TR = 0.5, iter_lim = 5, eps = 0.1)

    env2, mod2 = ExaAdmm.solve_sqp_ipopt(case; TR = 0.5, iter_lim = 5, eps = 0.1)
