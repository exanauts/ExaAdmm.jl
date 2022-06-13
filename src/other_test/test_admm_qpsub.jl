using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

@testset "Testing branch kernel updates" begin

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
data = mod.grid_data

env.params.scale = 1e-4
env.params.initial_beta = 1e3
env.params.beta = 1e3

@inbounds begin 
    #fix random seed and generate SQP param for mod
    Random.seed!(20)

    ls_same = rand() #keep ls and us same for all as simplification
    us_same = ls_same + 10
    
    for i = 1: data.nline
        Hs = rand(6,6) 
        Hs = Hs * transpose(Hs) #make symmetric, inherit structure Hessian
        mod.Hs[6*(i-1)+1:6*i,1:6] .= Hs

        #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment in ipopt benchmark
        LH_1h = zeros(4) #LH * x = RH
        mod.LH_1h[i,:] .= LH_1h
        RH_1h = 0.0 
        mod.RH_1h[i] = RH_1h

        LH_1i = zeros(4) #Lf * x = RH
        mod.LH_1i[i,:] .= LH_1h
        RH_1i = 0.0
        mod.RH_1i[i] = RH_1i

        #inherit structure line limit constraint (overleaf)
        LH_1j = rand(2)
        mod.LH_1j[i,:] .= LH_1j
        RH_1j = rand() + 100 
        mod.RH_1j[i] = RH_1j

        LH_1k = rand(2)
        mod.LH_1k[i,:] .= LH_1k
        RH_1k = rand() + 100 
        mod.RH_1k[i] = RH_1k
        
        #inherit structure SQP for line variable bound
        mod.ls[i,:] .= ls_same  
        mod.us[i,:] .= us_same
    end






#check value 
println(data.pgmax)




# generate full qpsub with mod and solve by ipopt 
model = JuMP.Model(Ipopt.Optimizer)
set_silent(model)
# variables 
@variable(model, pg[1:data.ngen])
@variable(model, qg[1:data.ngen])
@variable(model, line_var[1:6,1:data.nline]) #w_ijR, w_ijI, w_i, w_j, theta_i, theta_j

@variable(model, line_fl[1:4,1:data.nline]) #p_ij, q_ij, p_ji, q_ji 

@variable(model, bus_w[1:data.nbus]) #for consensus
@variable(model, bus_theta[1:data.nbus]) #for consensus 

# objective (ignore constant in generation objective)
@objective(model, Min, sum(data.c2[g]*pg[g]^2 + data.c1[g]*pg[g] for g=1:data.ngen) +
    sum(0.5*dot(line_var[:,l],mod.Hs[6*(l-1)+1:6*l,1:6],line_var[:,l]) for l=1:data.nline) )

# generator constraint
@constraint(model, [g=1:data.ngen], data.pgmin[g] <= pg[g] <= data.pgmax[g])
@constraint(model, [g=1:data.ngen], data.qgmin[g] <= qg[g] <= data.qgmax[g])

# bus constraint

# line constraint (1h 1i igonred)
@constraint(model, [l=1:data.nline], mod.ls[l,:] .<= line_var[:,l] .<= mod.us[l,:]) #lower and upper bounds
@constraint(model, [l=1:data.nline], mod.LH_1j[l,1] * line_fl[1,l] + mod.LH_1j[l,2] * line_fl[2,l] <= mod.RH_1j[l])   #1j
@constraint(model, [l=1:data.nline], mod.LH_1k[l,1] * line_fl[3,l] + mod.LH_1k[l,2] * line_fl[4,l] <= mod.RH_1k[l])   #LH_1k

for l = data.nline #match line_fl with line_var
    supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
    -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
    data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
    -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
    @constraint(model, supY * line_var[:,l] .== line_fl[:,l])
end


# coupling constraint for consensus 

optimize!(model)








# solve full qpsub with admm 



end #inbounds
end #@testset