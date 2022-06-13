using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

@testset "Testing x,xbar,z,l,lz,residual update" begin

INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "mp_demand")

case = joinpath(INSTANCES_DIR, "case9.m")
rho_pq = 4e2; rho_va = 4e4
atol = 1e-6; verbose=0

# Initialize an qpsub model with default options.
T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
mod = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env)
sol = mod.solution
par = env.params

env.params.scale = 1e-4
env.params.initial_beta = 1e3
env.params.beta = 1e3

@inbounds begin 
    #fix random seed and generate SQP param for mod
    Random.seed!(20);
    for i = 1: mod.grid_data.nline
        Hs = rand(6,6) 
        Hs = Hs * transpose(Hs) #make symmetric, inherit structure Hessian
        mod.Hs[6*(i-1)+1:6*i,1:6] .= Hs

        #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment for testing 
        LH_1h = zeros(4) #LH * x = RH
        mod.LH_1h[i,:] .= LH_1h
        RH_1h = 0.0 
        mod.RH_1h[i] = RH_1h

        LH_1i = zeros(4) #Lf * x = RH
        mod.LH_1i[i,:] .= LH_1h
        RH_1i = 0.0
        mod.RH_1i[i] = RH_1i

        #inherit structure line limit 
        LH_1j = rand(2)
        mod.LH_1j[i,:] .= LH_1j
        RH_1j = sum(rand(1)) + 100 #reduce dim
        mod.RH_1j[i] = RH_1j

        LH_1k = rand(2)
        mod.LH_1k[i,:] .= LH_1k
        RH_1k = sum(rand(1)) + 100 #reduce dim
        mod.RH_1k[i] = RH_1k
        #inherit structure bound
        ls = rand(6)
        mod.ls[i,:] .= ls
        us = ls .+ 10 
        mod.us[i,:] .= us
    end
end #inbounds









# We'll perform just one ADMM iteration. Do some pre-steps first.
println("prestep starts")
ExaAdmm.admm_increment_outer(env, mod)
ExaAdmm.admm_outer_prestep(env, mod)
ExaAdmm.admm_increment_reset_inner(env, mod)
ExaAdmm.admm_increment_inner(env, mod)
ExaAdmm.admm_inner_prestep(env, mod)

println("update x start")
ExaAdmm.admm_update_x(env, mod)

println("update x_bar start")
ExaAdmm.admm_update_xbar(env, mod)

println("update z start")
ExaAdmm.admm_update_z(env, mod)

println("update λ start")
ExaAdmm.admm_update_l(env, mod)

println("update residual start")
ExaAdmm.admm_update_residual(env, mod)

println("update λz start")
ExaAdmm.admm_update_lz(env, mod)



end #@testset