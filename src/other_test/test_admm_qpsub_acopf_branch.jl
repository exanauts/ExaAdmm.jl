using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

# @testset "branch ALM vs IPOPT" begin

INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "matpower")


fix_line = false
case = joinpath(INSTANCES_DIR, "case9.m")
# case = joinpath(INSTANCES_DIR, "case30.m")
# case = joinpath(INSTANCES_DIR, "case118.m"); fix_line = true  
# case = joinpath(INSTANCES_DIR, "case300.m")
# case = joinpath(INSTANCES_DIR, "case1354pegase.m")
# case = joinpath(INSTANCES_DIR, "case2869pegase.m")

rho_pq = 20.0 #for two level 
rho_va = 20.0 #for two level
initial_beta = 100000.0 #for two level
verbose=0
atol = 1e-4

# Initialize an qpsub model with default options as shell for qpsub.
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}

    env1 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
    mod1 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env1)
    sol = mod1.solution
    par = env1.params
    data = mod1.grid_data
#


# generating reference point from disturbed acopf
@inbounds begin
    distQ = 1.0
    distL = 1.0

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

    @variable(model, bus_w[b = 1:data.nbus], start = 1.0) #for consensus
    @variable(model, bus_theta[b = 1:data.nbus], start = 0.0) #for consensus 




    # objective (ignore constant in generation objective)
    @objective(model, Min, sum( distQ*data.c2[g]*(pg[g]*data.baseMVA)^2 + distL*data.c1[g]*pg[g]*data.baseMVA + data.c0[g] for g=1:data.ngen))




    # generator constraint
    @constraint(model, [g=1:data.ngen], pg[g] <= data.pgmax[g])
    @constraint(model, [g=1:data.ngen], qg[g] <= data.qgmax[g])
    @constraint(model, [g=1:data.ngen], data.pgmin[g] <= pg[g] )
    @constraint(model, [g=1:data.ngen], data.qgmin[g] <= qg[g] )

    if !fix_line
        @constraint(model, [l=1:data.nline], line_fl[1,l]^2 + line_fl[2,l]^2 <= data.rateA[l])
        @constraint(model, [l=1:data.nline], line_fl[3,l]^2 + line_fl[4,l]^2 <= data.rateA[l])
    end

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

    #voltage and angle bound
    for l = 1:data.nline
        @constraint(model, line_var[3,l] >= data.FrVmBound[1+2*(l-1)]^2) #wi
        @constraint(model, line_var[3,l] <= data.FrVmBound[2*l]^2) #wi
        @constraint(model, line_var[4,l] >= data.ToVmBound[1+2*(l-1)]^2) #wj
        @constraint(model, line_var[4,l] <= data.ToVmBound[2*l]^2) #wj

        @constraint(model, line_var[5,l] >= data.FrVaBound[1+2*(l-1)]) #ti
        @constraint(model, line_var[5,l] <= data.FrVaBound[2*l]) #ti
        @constraint(model, line_var[6,l] >= data.ToVaBound[1+2*(l-1)]) #tj
        @constraint(model, line_var[6,l] <= data.ToVaBound[2*l]) #tj
    end

    # line constraint 
    @NLconstraint(model, [l=1:data.nline], (line_var[1,l])^2 + (line_var[2,l])^2 == line_var[3,l]*line_var[4,l] ) #wR^2 + wI^2 = wiwj
    @NLconstraint(model, [l=1:data.nline], line_var[1,l] * sin(line_var[5,l] - line_var[6,l]) == line_var[2,l] * cos(line_var[5,l] - line_var[6,l]))

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

end #@inbounds


# save variable to Hs, 1h, 1j, 1i, 1k, new bound, new cost 
@inbounds begin 
    pi_14 = -ones(4,data.nline) #set multiplier for the hessian evaluation 14h 14i 14j 14k
    is_Hs_sym = zeros(data.nline) #is Hs symmetric
    is_Hs_PSD = zeros(data.nline) #is Hs positive semidefinite 


    #gen bound
    data.pgmax .-= value.(pg)  
    data.pgmin .-= value.(pg)
    data.qgmax .-= value.(qg)
    data.qgmin .-= value.(qg)

    #new cost coeff
    data.c1 .+= 2*data.c2.*value.(pg)
    
    #w theta bound
    for l = 1: data.nline
        mod1.ls[l,1] = -2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijR lb
        mod1.us[l,1] = 2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijR ub
        mod1.ls[l,2] = -2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijI lb
        mod1.us[l,2] = 2*data.FrVmBound[2*l]*data.ToVmBound[2*l] #wijI ub

        mod1.ls[l,3] = data.FrVmBound[1+2*(l-1)]^2 - value.(line_var)[3,l] #wi lb
        mod1.us[l,3] = data.FrVmBound[2*l]^2 - value.(line_var)[3,l] #wi ub
        mod1.ls[l,4] = data.ToVmBound[1+2*(l-1)]^2 - value.(line_var)[4,l] #wj lb
        mod1.us[l,4] = data.ToVmBound[2*l]^2 - value.(line_var)[4,l] #wj ub

        mod1.ls[l,5] = data.FrVaBound[1+2*(l-1)] - value.(line_var)[5,l] #ti lb
        mod1.us[l,5] = data.FrVaBound[2*l] - value.(line_var)[5,l] #ti ub
        mod1.ls[l,6] = data.ToVaBound[1+2*(l-1)] - value.(line_var)[6,l] #tj lb
        mod1.us[l,6] = data.ToVaBound[2*l] - value.(line_var)[6,l] #tj ub
    end

    for b = 1:data.nbus
        data.Pd[b] = data.baseMVA * (data.Pd[b]/data.baseMVA - (value.(pgb)[b] - value.(pft)[b] - value.(ptf)[b] - data.YshR[b]*value.(bus_w)[b]))
        data.Qd[b] = data.baseMVA * (data.Qd[b]/data.baseMVA - (value.(qgb)[b] - value.(qft)[b] - value.(qtf)[b] + data.YshI[b]*value.(bus_w)[b]))
    end

    for l = 1: data.nline
        #Hs:(6,6) #w_ijR, w_ijI, w_i, w_j, theta_i, theta_j
        
        Hs = zeros(6,6)

        Hs_14h = zeros(6,6)
        Hs_14h[1,1] = 2*pi_14[1,l]
        Hs_14h[2,2] = 2*pi_14[1,l] 
        Hs_14h[3,4] = -pi_14[1,l] 
        Hs_14h[4,3] = -pi_14[1,l] 

        Hs_14i = zeros(6,6)
        cons_1 = pi_14[2,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l])
        cons_2 = pi_14[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])
        cons_3 = pi_14[2,l]*(-value.(line_var)[1,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l]) +  value.(line_var)[1,2]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]))
    
        Hs_14i[1,5] = cons_1 #wijR theta_i
        Hs_14i[5,1] = cons_1 #wijR theta_i
        Hs_14i[1,6] = -cons_1 #wijR theta_j
        Hs_14i[6,1] = -cons_1 #wijR theta_j 

        Hs_14i[2,5] = cons_2 #wijR theta_i
        Hs_14i[5,2] = cons_2 #wijR theta_i
        Hs_14i[2,6] = -cons_2 #wijR theta_j
        Hs_14i[6,2] = -cons_2 #wijR theta_j 

        Hs_14i[5,5] = cons_3 #thetai thetai
        Hs_14i[6,6] = cons_3 #thetaj thetaj
        Hs_14i[5,6] = -cons_3 #thetai thetaj
        Hs_14i[6,5] = -cons_3 #thetaj thetai 
        
        supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
            -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
            data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
            -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
        Hs_14j = -2*pi_14[3,l]*(supY[1,:]*transpose(supY[1,:]) + supY[2,:]*transpose(supY[2,:]) )
        Hs_14k = -2*pi_14[4,l]*(supY[3,:]*transpose(supY[3,:]) + supY[4,:]*transpose(supY[4,:]) )
        Hs .= Hs_14h + Hs_14i + Hs_14j + Hs_14k + UniformScaling(4)#with multiplier pi_14 #! fix to psd for now
        mod1.Hs[6*(l-1)+1:6*l,1:6] .= Hs

        is_Hs_sym[l] = maximum(abs.(Hs - transpose(Hs)))
        eival, eivec = eigen(Hs)
        is_Hs_PSD[l] = minimum(eival)

        #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment in ipopt benchmark
        LH_1h = [2*value.(line_var[1,l]), 2*value.(line_var[2,l]), -value.(line_var[4,l]), -value.(line_var[3,l])] #LH * x = RH
        mod1.LH_1h[l,:] .= LH_1h
        RH_1h = -(value.(line_var)[1,l])^2 - (value.(line_var)[2,l])^2 + value.(line_var)[3,l]*value.(line_var)[4,l] 
        mod1.RH_1h[l] = RH_1h

        LH_1i = [sin(value.(line_var)[5,l] - value.(line_var)[6,l]), -cos(value.(line_var)[5,l] - value.(line_var)[6,l]), 
        value.(line_var)[1,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]) +  value.(line_var)[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l]),
        -value.(line_var)[1,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l]) -  value.(line_var)[2,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])] #Lf * x = RH
        mod1.LH_1i[l,:] .= LH_1i
        RH_1i = -value.(line_var)[1,l]*sin(value.(line_var)[5,l] - value.(line_var)[6,l])  +  value.(line_var)[2,l]*cos(value.(line_var)[5,l] - value.(line_var)[6,l])
        mod1.RH_1i[l] = RH_1i

        #inherit structure line limit constraint (overleaf)
        LH_1j = [2*value.(line_fl)[1,l], 2*value.(line_fl)[2,l]] #rand(2)
        mod1.LH_1j[l,:] .= LH_1j
        RH_1j = -((value.(line_fl)[1,l])^2 + (value.(line_fl)[2,l])^2 - data.rateA[l]) 
        mod1.RH_1j[l] = RH_1j

        LH_1k = [2*value.(line_fl[3,l]), 2*value.(line_fl[4,l])] #zeros(2) #rand(2)
        mod1.LH_1k[l,:] .= LH_1k
        RH_1k = -((value.(line_fl)[3,l])^2 + (value.(line_fl)[4,l])^2 - data.rateA[l]) 
        mod1.RH_1k[l] = RH_1k
    end
end #inbound 


# ipopt solve admm problem
use_ipopt = true
if use_ipopt

    # generate full qpsub with mod and solve by ipopt 
    model2 = JuMP.Model(Ipopt.Optimizer)
    set_silent(model2)
    
    
    
    # variables 
    @variable(model2, pg[1:data.ngen])
    @variable(model2, qg[1:data.ngen])
    @variable(model2, line_var[1:6,1:data.nline]) #w_ijR, w_ijI, w_i, w_j, theta_i, theta_j
    
    @variable(model2, line_fl[1:4,1:data.nline]) #p_ij, q_ij, p_ji, q_ji 
    
    @variable(model2, pft[1:data.nbus]) #sum pij over j in B_i (frombus)
    @variable(model2, ptf[1:data.nbus]) #sum pij over j in B_i (tobus)
    @variable(model2, pgb[1:data.nbus]) #sum pg over g in G_i
    
    @variable(model2, qft[1:data.nbus]) #sum qij over j in B_i (frombus)
    @variable(model2, qtf[1:data.nbus]) #sum qij over j in B_i (tobus)
    @variable(model2, qgb[1:data.nbus]) #sum qg over g in G_i
    
    @variable(model2, bus_w[1:data.nbus]) #for consensus
    @variable(model2, bus_theta[1:data.nbus]) #for consensus 
    
    
    
    
    # objective (ignore constant in generation objective)
    @objective(model2, Min, sum(data.c2[g]*(pg[g]*data.baseMVA)^2 + data.c1[g]*pg[g]*data.baseMVA for g=1:data.ngen) +
        sum(0.5*dot(line_var[:,l],mod1.Hs[6*(l-1)+1:6*l,1:6],line_var[:,l]) for l=1:data.nline) )
    
    
    
    
    # generator constraint
    @constraint(model2, [g=1:data.ngen], pg[g] <= data.pgmax[g])
    @constraint(model2, [g=1:data.ngen], qg[g] <= data.qgmax[g])
    @constraint(model2, [g=1:data.ngen], data.pgmin[g] <= pg[g] )
    @constraint(model2, [g=1:data.ngen], data.qgmin[g] <= qg[g] )
    
    
    
    
    # bus constraint: power balance
    # pd
    for b = 1:data.nbus
        if data.FrStart[b] < data.FrStart[b+1]
            @constraint(model2, pft[b] == sum( line_fl[1,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1))
        else
            @constraint(model2, pft[b] == 0)
        end
    
        if data.ToStart[b] < data.ToStart[b+1]
            @constraint(model2, ptf[b] == sum( line_fl[3,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1))
        else
            @constraint(model2, ptf[b] == 0)
        end
    
        if data.GenStart[b] < data.GenStart[b+1]
            @constraint(model2, pgb[b] == sum( pg[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1))
        else
            @constraint(model2, pgb[b] == 0)
        end
    
        @constraint(model2, pgb[b] - pft[b] - ptf[b] - data.YshR[b]*bus_w[b] == data.Pd[b]/data.baseMVA) 
    end
    
    #qd
    for b = 1:data.nbus
        if data.FrStart[b] < data.FrStart[b+1]
            @constraint(model2, qft[b] == sum( line_fl[2,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1))
        else
            @constraint(model2, qft[b] == 0)
        end
    
        if data.ToStart[b] < data.ToStart[b+1]
            @constraint(model2, qtf[b] == sum( line_fl[4,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1))
        else
            @constraint(model2, qtf[b] == 0)
        end
    
        if data.GenStart[b] < data.GenStart[b+1]
            @constraint(model2, qgb[b] == sum( qg[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1))
        else
            @constraint(model2, qgb[b] == 0)
        end
    
        @constraint(model2, qgb[b] - qft[b] - qtf[b] + data.YshI[b]*bus_w[b] == data.Qd[b]/data.baseMVA) 
    end
    
    
    
    
    # line constraint (1h 1i igonred)
    @constraint(model2, [l=1:data.nline], mod1.ls[l,:] .<= line_var[:,l] .<= mod1.us[l,:]) #lower and upper bounds
    @constraint(model2, [l=1:data.nline], mod1.LH_1j[l,1] * line_fl[1,l] + mod1.LH_1j[l,2] * line_fl[2,l] <= mod1.RH_1j[l])   #1j
    @constraint(model2, [l=1:data.nline], mod1.LH_1k[l,1] * line_fl[3,l] + mod1.LH_1k[l,2] * line_fl[4,l] <= mod1.RH_1k[l])   #1k

    @constraint(model2, [l=1:data.nline], sum(mod1.LH_1h[l,i] * line_var[i,l] for i=1:4) == mod1.RH_1h[l])   #1h
    @constraint(model2, [l=1:data.nline], mod1.LH_1i[l,1] * line_var[1,l] + mod1.LH_1i[l,2] * line_var[2,l] + mod1.LH_1i[l,3] * line_var[5,l] + mod1.LH_1i[l,4] * line_var[6,l]  == mod1.RH_1i[l])   #1i
    


    for l = 1:data.nline #match line_fl with line_var
        supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
        -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
        data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
        -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
        @constraint(model2, supY * line_var[:,l] .== line_fl[:,l])
    end
    
    
    
    
    
    # coupling constraint for consensus 
    for b = 1:data.nbus
        for k = data.FrStart[b]:data.FrStart[b+1]-1
            @constraint(model2, bus_w[b] == line_var[3, data.FrIdx[k]]) #wi(ij)
            @constraint(model2, bus_theta[b] == line_var[5, data.FrIdx[k]]) #ti(ij)
        end
        for k = data.ToStart[b]:data.ToStart[b+1]-1
            @constraint(model2, bus_w[b] == line_var[4, data.ToIdx[k]]) #wj(ji)
            @constraint(model2, bus_theta[b] == line_var[6, data.ToIdx[k]]) #tj(ji)
        end
    end
    
    
    optimize!(model2)
    
    println(termination_status(model2))

end #if use_ipopt 

# test branch accuracy
res = zeros(data.nline)
res2 = zeros(data.nline)
res3 = zeros(data.nline)
@inbounds begin
    for i = 1: data.nline #mod.grid_data.nline
        
        shift_idx = mod1.line_start + 8*(i-1)

        supY = [mod1.grid_data.YftR[i]  mod1.grid_data.YftI[i]  mod1.grid_data.YffR[i] 0 0 0;
        -mod1.grid_data.YftI[i]  mod1.grid_data.YftR[i]  -mod1.grid_data.YffI[i] 0 0 0;
        mod1.grid_data.YtfR[i]  -mod1.grid_data.YtfI[i]  0  mod1.grid_data.YttR[i] 0 0;
        -mod1.grid_data.YtfI[i]  -mod1.grid_data.YtfR[i] 0  -mod1.grid_data.YttI[i] 0 0]
        
        # print("1: ",mod.Hs[6*(i-1)+1:6*i,1:6])
        A_ipopt = ExaAdmm.eval_A_branch_kernel_cpu_qpsub(mod1.Hs[6*(i-1)+1:6*i,1:6], sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod1.grid_data.YffR[i], mod1.grid_data.YffI[i],
        mod1.grid_data.YftR[i], mod1.grid_data.YftI[i],
        mod1.grid_data.YttR[i], mod1.grid_data.YttI[i],
        mod1.grid_data.YtfR[i], mod1.grid_data.YtfI[i])
        # print("2: ",mod.Hs[6*(i-1)+1:6*i,1:6])

        b_ipopt = ExaAdmm.eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod1.grid_data.YffR[i], mod1.grid_data.YffI[i],
        mod1.grid_data.YftR[i], mod1.grid_data.YftI[i],
        mod1.grid_data.YttR[i], mod1.grid_data.YttI[i],
        mod1.grid_data.YtfR[i], mod1.grid_data.YtfI[i])
        # println()
        # print("3 ",mod.Hs[6*(i-1)+1:6*i,1:6])

        l_ipopt = mod1.ls[i,:]
        u_ipopt = mod1.us[i,:]
        
        # call Ipopt check QP
        local model = JuMP.Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, l_ipopt[k]<= x[k=1:6] <=u_ipopt[k])
        @objective(model, Min, 0.5 * dot(x, A_ipopt, x) + dot(b_ipopt, x))
        @constraint(model, eq1h, dot(mod1.LH_1h[i,:], x[1:4]) == mod1.RH_1h[i])
        @constraint(model, eq1i, dot(mod1.LH_1i[i,:], [x[1:2];x[5:6]]) == mod1.RH_1i[i])
        @constraint(model, sij, mod1.LH_1j[i,1] * dot(supY[1,:],x) + mod1.LH_1j[i,2] * dot(supY[2,:],x) <= mod1.RH_1j[i])
        @constraint(model, sji, mod1.LH_1k[i,1] * dot(supY[3,:],x) + mod1.LH_1k[i,2] * dot(supY[4,:],x) <= mod1.RH_1k[i])
        optimize!(model)
        x_ipopt1 = value.(x)
        # println("objective = ", objective_value(model), " with solution = ",x_ipopt1)

        # call Ipopt check ALM
        local model2 = JuMP.Model(Ipopt.Optimizer)
        set_silent(model2)
        @variable(model2, l_ipopt[k]<= x[k=1:6] <=u_ipopt[k])
        @objective(model2, Min, 0.5*dot(x,mod1.Hs[6*(i-1)+1:6*i,1:6],x) + 
                    sol.l_curr[shift_idx]*dot(supY[1,:],x) + sol.l_curr[shift_idx + 1]*dot(supY[2,:],x) +
                    sol.l_curr[shift_idx + 2]*dot(supY[3,:],x) + sol.l_curr[shift_idx + 3]*dot(supY[4,:],x) +
                    sol.l_curr[shift_idx + 4]*x[3] + sol.l_curr[shift_idx + 5]*x[4] + sol.l_curr[shift_idx + 6]*x[5] + sol.l_curr[shift_idx + 7]*x[6] +
                    0.5*sol.rho[shift_idx]*(dot(supY[1,:],x) - sol.v_curr[shift_idx] + sol.z_curr[shift_idx])^2 +
                    0.5*sol.rho[shift_idx + 1]*(dot(supY[2,:],x) - sol.v_curr[shift_idx + 1] + sol.z_curr[shift_idx + 1])^2 +
                    0.5*sol.rho[shift_idx + 2]*(dot(supY[3,:],x) - sol.v_curr[shift_idx + 2] + sol.z_curr[shift_idx + 2])^2 +
                    0.5*sol.rho[shift_idx + 3]*(dot(supY[4,:],x) - sol.v_curr[shift_idx + 3] + sol.z_curr[shift_idx + 3])^2 +
                    0.5*sol.rho[shift_idx + 4]*(x[3]-sol.v_curr[shift_idx + 4] + sol.z_curr[shift_idx + 4])^2 + 
                    0.5*sol.rho[shift_idx + 5]*(x[4]-sol.v_curr[shift_idx + 5] + sol.z_curr[shift_idx + 5])^2 +
                    0.5*sol.rho[shift_idx + 6]*(x[5]-sol.v_curr[shift_idx + 6] + sol.z_curr[shift_idx + 6])^2 +
                    0.5*sol.rho[shift_idx + 7]*(x[6]-sol.v_curr[shift_idx + 7] + sol.z_curr[shift_idx + 7])^2 )
        
                    @constraint(model2, eq1h, dot(mod1.LH_1h[i,:], x[1:4]) == mod1.RH_1h[i])
                    @constraint(model2, eq1i, dot(mod1.LH_1i[i,:], [x[1:2];x[5:6]]) == mod1.RH_1i[i])
                    @constraint(model2, sij, mod1.LH_1j[i,1] * dot(supY[1,:],x) + mod1.LH_1j[i,2] * dot(supY[2,:],x) <= mod1.RH_1j[i])
                    @constraint(model2, sji, mod1.LH_1k[i,1] * dot(supY[3,:],x) + mod1.LH_1k[i,2] * dot(supY[4,:],x) <= mod1.RH_1k[i])
        optimize!(model2)
        x_ipopt2 = value.(x)
        # println("objective = ", objective_value(model2), " with solution = ",x_ipopt2)

        # println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_prev[pij]
        tronx, tronf = ExaAdmm.auglag_Ab_linelimit_two_level_alternative_qpsub_ij(1, par.max_auglag, par.mu_max, 1e-4, A_ipopt, b_ipopt, mod1.ls[i,:], mod1.us[i,:], mod1.sqp_line, sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], mod1.qpsub_membuf,i,
        mod1.grid_data.YffR[i], mod1.grid_data.YffI[i],
        mod1.grid_data.YftR[i], mod1.grid_data.YftI[i],
        mod1.grid_data.YttR[i], mod1.grid_data.YttI[i],
        mod1.grid_data.YtfR[i], mod1.grid_data.YtfI[i],
        mod1.LH_1h[i,:], mod1.RH_1h[i], mod1.LH_1i[i,:], mod1.RH_1i[i], mod1.LH_1j[i,:], mod1.RH_1j[i], mod1.LH_1k[i,:], mod1.RH_1k[i])
        # println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_curr[pij]

        tronx2, tronf2 = ExaAdmm.auglag_Ab_linelimit_two_level_alternative_qpsub_ij_red(1, par.max_auglag, par.mu_max, 1e-4, A_ipopt, b_ipopt, mod1.ls[i,:], mod1.us[i,:], mod1.sqp_line, sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], mod1.qpsub_membuf,i,
        mod1.grid_data.YffR[i], mod1.grid_data.YffI[i],
        mod1.grid_data.YftR[i], mod1.grid_data.YftI[i],
        mod1.grid_data.YttR[i], mod1.grid_data.YttI[i],
        mod1.grid_data.YtfR[i], mod1.grid_data.YtfI[i],
        mod1.LH_1h[i,:], mod1.RH_1h[i], mod1.LH_1i[i,:], mod1.RH_1i[i], mod1.LH_1j[i,:], mod1.RH_1j[i], mod1.LH_1k[i,:], mod1.RH_1k[i])

        res[i] = norm(tronx[3:8] - x_ipopt1, Inf)
        res2[i] = norm(tronx[3:8] - x_ipopt2, Inf)
        res3[i] = norm(tronx[3:8] - tronx2[3:8], Inf)

    end

end

# @test norm(res, Inf) <= atol
# @test norm(res2, Inf) <= atol

# end #test 