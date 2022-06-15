using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

use_ipopt = true 



INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "mp_demand")

case = joinpath(INSTANCES_DIR, "case9.m")
# case = joinpath(INSTANCES_DIR, "case118.m")
rho_pq = 20.0 
rho_va = 20.0
# scale = 1
initial_beta = 100000.0
atol = 1e-6
verbose=1


# Initialize an qpsub model with default options.
T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}

env1 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
mod1 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env1)
sol = mod1.solution
par = env1.params
data = mod1.grid_data


@inbounds begin 
    #fix random seed and generate SQP param for mod
    Random.seed!(1)

    ls_same = -1.5 #keep ls and us same for all as simplification
    us_same = 1.5
    
    for i = 1: data.nline
        Hs = rand(6,6) 
        Hs = Hs * transpose(Hs) #make symmetric, inherit structure Hessian
        mod1.Hs[6*(i-1)+1:6*i,1:6] .= 100*Hs

        #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment in ipopt benchmark
        LH_1h = zeros(4) #LH * x = RH
        mod1.LH_1h[i,:] .= LH_1h
        RH_1h = 0.0 
        mod1.RH_1h[i] = RH_1h

        LH_1i = zeros(4) #Lf * x = RH
        mod1.LH_1i[i,:] .= LH_1h
        RH_1i = 0.0
        mod1.RH_1i[i] = RH_1i

        #inherit structure line limit constraint (overleaf)
        LH_1j = zeros(2) #rand(2)
        mod1.LH_1j[i,:] .= LH_1j
        RH_1j = 0 #rand() + 200 
        mod1.RH_1j[i] = RH_1j

        LH_1k = zeros(2) #rand(2)
        mod1.LH_1k[i,:] .= LH_1k
        RH_1k = 0  #rand() + 200 
        mod1.RH_1k[i] = RH_1k
        
        #inherit structure SQP for line variable bound
        mod1.ls[i,:] .= ls_same  
        mod1.us[i,:] .= us_same
    end






#check value 
# println(data.pgmax)


if use_ipopt

# generate full qpsub with mod and solve by ipopt 
model = JuMP.Model(Ipopt.Optimizer)
set_silent(model)



# variables 
@variable(model, pg[1:data.ngen])
@variable(model, qg[1:data.ngen])
@variable(model, line_var[1:6,1:data.nline]) #w_ijR, w_ijI, w_i, w_j, theta_i, theta_j

@variable(model, line_fl[1:4,1:data.nline]) #p_ij, q_ij, p_ji, q_ji 

@variable(model, pft[1:data.nbus]) #sum pij over j in B_i (frombus)
@variable(model, ptf[1:data.nbus]) #sum pij over j in B_i (tobus)
@variable(model, pgb[1:data.nbus]) #sum pg over g in G_i

@variable(model, qft[1:data.nbus]) #sum qij over j in B_i (frombus)
@variable(model, qtf[1:data.nbus]) #sum qij over j in B_i (tobus)
@variable(model, qgb[1:data.nbus]) #sum qg over g in G_i

@variable(model, bus_w[1:data.nbus]) #for consensus
@variable(model, bus_theta[1:data.nbus]) #for consensus 




# objective (ignore constant in generation objective)
@objective(model, Min, sum(data.c2[g]*(pg[g]*data.baseMVA)^2 + data.c1[g]*pg[g]*data.baseMVA + data.c0[g] for g=1:data.ngen) +
    sum(0.5*dot(line_var[:,l],mod1.Hs[6*(l-1)+1:6*l,1:6],line_var[:,l]) for l=1:data.nline) )




# generator constraint
@constraint(model, [g=1:data.ngen], pg[g] <= data.pgmax[g])
@constraint(model, [g=1:data.ngen], qg[g] <= data.qgmax[g])
@constraint(model, [g=1:data.ngen], data.pgmin[g] <= pg[g] )
@constraint(model, [g=1:data.ngen], data.qgmin[g] <= qg[g] )




# bus constraint: power balance
# pd
for b = 1:data.nbus
    if data.FrStart[b] < data.FrStart[b+1]
        @constraint(model, pft[b] == sum( line_fl[1,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1))
    else
        @constraint(model, pft[b] == 0)
    end

    if data.ToStart[b] < data.ToStart[b+1]
        @constraint(model, ptf[b] == sum( line_fl[3,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1))
    else
        @constraint(model, ptf[b] == 0)
    end

    if data.GenStart[b] < data.GenStart[b+1]
        @constraint(model, pgb[b] == sum( pg[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1))
    else
        @constraint(model, pgb[b] == 0)
    end

    @constraint(model, pgb[b] - pft[b] - ptf[b] - data.YshR[b]*bus_w[b] == data.Pd[b]/data.baseMVA) 
end

#qd
for b = 1:data.nbus
    if data.FrStart[b] < data.FrStart[b+1]
        @constraint(model, qft[b] == sum( line_fl[2,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1))
    else
        @constraint(model, qft[b] == 0)
    end

    if data.ToStart[b] < data.ToStart[b+1]
        @constraint(model, qtf[b] == sum( line_fl[4,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1))
    else
        @constraint(model, qtf[b] == 0)
    end

    if data.GenStart[b] < data.GenStart[b+1]
        @constraint(model, qgb[b] == sum( qg[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1))
    else
        @constraint(model, qgb[b] == 0)
    end

    @constraint(model, qgb[b] - qft[b] - qtf[b] + data.YshI[b]*bus_w[b] == data.Qd[b]/data.baseMVA) 
end




# line constraint (1h 1i igonred)
@constraint(model, [l=1:data.nline], mod1.ls[l,:] .<= line_var[:,l] .<= mod1.us[l,:]) #lower and upper bounds
# @constraint(model, [l=1:data.nline], mod1.LH_1j[l,1] * line_fl[1,l] + mod1.LH_1j[l,2] * line_fl[2,l] <= mod1.RH_1j[l])   #1j
# @constraint(model, [l=1:data.nline], mod1.LH_1k[l,1] * line_fl[3,l] + mod1.LH_1k[l,2] * line_fl[4,l] <= mod1.RH_1k[l])   #1k

for l = 1:data.nline #match line_fl with line_var
    supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
    -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
    data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
    -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
    @constraint(model, supY * line_var[:,l] .== line_fl[:,l])
end





# coupling constraint for consensus 
for b = 1:data.nbus
    for k = data.FrStart[b]:data.FrStart[b+1]-1
        @constraint(model, bus_w[b] == line_var[3, data.FrIdx[k]]) #wi(ij)
        @constraint(model, bus_theta[b] == line_var[5, data.FrIdx[k]]) #ti(ij)
    end
    for k = data.ToStart[b]:data.ToStart[b+1]-1
        @constraint(model, bus_w[b] == line_var[4, data.ToIdx[k]]) #wj(ji)
        @constraint(model, bus_theta[b] == line_var[6, data.ToIdx[k]]) #tj(ji)
    end
end


optimize!(model)

println(termination_status(model))

# println([value.(pg); value.(qg)])
# println(value.(line_var))
# println(value.(line_fl))

end #if use_ipopt 

# solve full qpsub with admm 


end #inbounds


env2, mod2 = ExaAdmm.solve_qpsub(case, mod1.Hs, mod1.LH_1h, mod1.RH_1h,
mod1.LH_1i, mod1.RH_1i, mod1.LH_1j, mod1.RH_1j, mod1.LH_1k, mod1.RH_1k, mod1.ls, mod1.us, initial_beta; 
outer_iterlim=3000, inner_iterlim=1, scale = 1.0, obj_scale = 1, rho_pq=20.0, rho_va=20.0, verbose=1, outer_eps=2*1e-5, onelevel = true);

# # println(mod.info.status)

# sol_x_gen = mod2.solution.u_curr[1:2*mod2.grid_data.ngen];

# sol_xbar_gen = mod2.solution.v_curr[1:2*mod2.grid_data.ngen];

# sol_x_line = reshape(mod2.solution.u_curr[mod2.line_start:end],(8,mod2.grid_data.nline));

# sol_xbar_line = reshape(mod2.solution.v_curr[mod2.line_start:end],(8,mod2.grid_data.nline));

