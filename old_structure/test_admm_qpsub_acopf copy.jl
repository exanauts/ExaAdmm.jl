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
verbose=1


# Initialize an qpsub model with default options as shell for qpsub.
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}

    env1 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
    mod1 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env1)
    sol = mod1.solution
    par = env1.params
    data = mod1.grid_data
#
    ExaAdmm.init_solution_sqp!(mod1, mod1.solution)

@inbounds begin
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

    @variable(model, bus_w[b = 1:data.nbus]) #for consensus
    @variable(model, bus_theta[b = 1:data.nbus]) #for consensus 




    # objective: ||x-x_k||_2^2
    # @objective(model, Min, sum( (pg[i] - mod1.pg_sol[i])^2 + (qg[i] - mod1.qg_sol[i])^2 for i = 1:data.ngen) 
    #             + sum( (bus_w[i] - mod1.w_sol[i])^2 + (bus_theta[i] - mod1.theta_sol[i])^2 for i = 1:data.nbus) 
    #             + sum( (line_var[i,j] - mod1.line_var[i,j])^2 for i = 1:6, j = 1:data.nline) 
    #             + sum( (line_fl[i,j] - mod1.line_fl[i,j])^2 for i = 1:4, j = 1:data.nline) 
    # )
    @objective(model, Min, sum( data.c2[g]*(pg[g]*data.baseMVA)^2 + data.c1[g]*pg[g]*data.baseMVA + data.c0[g] for g=1:data.ngen))

    # generator bound constraint
    @constraint(model, [g=1:data.ngen], pg[g] <= data.pgmax[g])
    @constraint(model, [g=1:data.ngen], qg[g] <= data.qgmax[g])
    @constraint(model, [g=1:data.ngen], data.pgmin[g] <= pg[g] )
    @constraint(model, [g=1:data.ngen], data.qgmin[g] <= qg[g] )

   
    @constraint(model, [l=1:data.nline], line_fl[1,l]^2 + line_fl[2,l]^2 <= data.rateA[l])
    @constraint(model, [l=1:data.nline], line_fl[3,l]^2 + line_fl[4,l]^2 <= data.rateA[l])
    @NLconstraint(model, [l=1:data.nline], (line_var[1,l])^2 + (line_var[2,l])^2 == line_var[3,l]*line_var[4,l] ) #wR^2 + wI^2 = wiwj
    @NLconstraint(model, [l=1:data.nline], line_var[1,l] * sin(line_var[5,l] - line_var[6,l]) == line_var[2,l] * cos(line_var[5,l] - line_var[6,l]))

    #line flow bound constraint 
    # @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[1,l] <= sqrt(data.rateA[l])) #pij
    # @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[2,l] <= sqrt(data.rateA[l])) #qij
    # @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[3,l] <= sqrt(data.rateA[l])) #pji
    # @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[4,l] <= sqrt(data.rateA[l])) #qji

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
        # @constraint(model, line_var[1,l] >= data.FrVmBound[1+2*(l-1)] * data.ToVmBound[1+2*(l-1)]) #wijR
        # @constraint(model, line_var[1,l] <= data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijR

        # @constraint(model, line_var[2,l] >= data.FrVmBound[1+2*(l-1)] * data.ToVmBound[1+2*(l-1)]) #wijI
        # @constraint(model, line_var[2,l] <= data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijI

        @constraint(model, line_var[3,l] >= data.FrVmBound[1+2*(l-1)]^2) #wi
        @constraint(model, line_var[3,l] <= data.FrVmBound[2*l]^2) #wi
        @constraint(model, line_var[4,l] >= data.ToVmBound[1+2*(l-1)]^2) #wj
        @constraint(model, line_var[4,l] <= data.ToVmBound[2*l]^2) #wj

        @constraint(model, line_var[5,l] >= data.FrVaBound[1+2*(l-1)]) #ti
        @constraint(model, line_var[5,l] <= data.FrVaBound[2*l]) #ti
        @constraint(model, line_var[6,l] >= data.ToVaBound[1+2*(l-1)]) #tj
        @constraint(model, line_var[6,l] <= data.ToVaBound[2*l]) #tj
    end

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