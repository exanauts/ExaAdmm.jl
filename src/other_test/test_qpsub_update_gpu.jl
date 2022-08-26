using Test
using LazyArtifacts
using LinearAlgebra
using Printf


using ExaAdmm
using Random
using JuMP
using Ipopt


# INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
# MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "matpower")

# @testset "Testing [x,xbar,l,res] updates and a solve for ACOPF" begin

rho_pq = 20.0 #for two level 
rho_va = 20.0 #for two level
initial_beta = 100000.0 #for two level
atol = 2e-6
verbose = 0
fix_line = false
use_gpu = false

case = "case9.m"

# Initialize an cpu qpsub model with default options as shell for qpsub.
 T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    
 env1 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
 mod1 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env1)
 sol = mod1.solution
 par = env1.params
 data = mod1.grid_data
#

    # generating reference point from disturbed acopf
    @inbounds begin
        distQ = 2.0
        distL = 3.0
    
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
        mod1.qpsub_pgmax .= data.pgmax - value.(pg)  
        mod1.qpsub_pgmin .= data.pgmin - value.(pg)
        mod1.qpsub_qgmax .= data.qgmax - value.(qg)
        mod1.qpsub_qgmin .= data.qgmin - value.(qg)
    
        #new cost coeff
        mod1.qpsub_c1 = data.c1 + 2*data.c2.*value.(pg)
        mod1.qpsub_c2 = data.c2
    
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
            mod1.qpsub_Pd[b] = data.baseMVA * (data.Pd[b]/data.baseMVA - (value.(pgb)[b] - value.(pft)[b] - value.(ptf)[b] - data.YshR[b]*value.(bus_w)[b]))
            mod1.qpsub_Qd[b] = data.baseMVA * (data.Qd[b]/data.baseMVA - (value.(qgb)[b] - value.(qft)[b] - value.(qtf)[b] + data.YshI[b]*value.(bus_w)[b]))
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
            @assert is_Hs_sym[l] <= 1e-6
            eival, eivec = eigen(Hs)
            is_Hs_PSD[l] = minimum(eival)
            @assert is_Hs_PSD[l] >= 0.0
    
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

    # initialize gpu_no
    if use_gpu
        using CUDA
        gpu_no = 1
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env2 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
    mod2 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env2)
    sol2 = mod2.solution
    par2 = env2.params
    data2 = mod2.grid_data
    info2 = mod2.info
    
    env2.params.scale = 1e-4
    env2.params.initial_beta = 1e3
    env2.params.beta = 1e3
    
    mod2.Hs = copy(mod1.Hs)
    mod2.LH_1h = copy(mod1.LH_1h)
    mod2.RH_1h = copy(mod1.RH_1h)
    mod2.LH_1i = copy(mod1.LH_1i)
    mod2.RH_1i = copy(mod1.RH_1i)
    mod2.LH_1j = copy(mod1.LH_1j)
    mod2.RH_1j = copy(mod1.RH_1j)
    mod2.LH_1k = copy(mod1.LH_1k)
    mod2.RH_1k = copy(mod1.RH_1k)
    mod2.ls = copy(mod1.ls)
    mod2.us = copy(mod1.us)
    
    mod2.qpsub_pgmax = copy(mod1.qpsub_pgmax)
    mod2.qpsub_pgmin = copy(mod1.qpsub_pgmin)
    mod2.qpsub_qgmax = copy(mod1.qpsub_qgmax)
    mod2.qpsub_qgmin = copy(mod1.qpsub_qgmin)

    mod2.qpsub_c1 = copy(mod1.qpsub_c1)
    mod2.qpsub_c2 = copy(mod1.qpsub_c2)
    mod2.qpsub_Pd = copy(mod1.qpsub_Pd)
    mod2.qpsub_Qd = copy(mod1.qpsub_Qd)

    ExaAdmm.init_solution!(mod2, mod2.solution, env2.initial_rho_pq, env2.initial_rho_va)
    
    info2.norm_z_prev = info2.norm_z_curr = 0
    par2.initial_beta = 0 
    par2.beta = 0
    sol2.lz .= 0
    sol2.z_curr .= 0
    sol2.z_prev .= 0
    par2.inner_iterlim = 1
    #? shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n) where n = 6
    par2.shmem_size = sizeof(Float64)*(16*mod2.n+4*mod2.n^2+178) + sizeof(Int)*(4*mod2.n)

    
    println()
    println()
    println("First iteration update comparison CPU vs GPU summary: ")
    println("prestep starts")
    ExaAdmm.admm_increment_outer(env2, mod2)
    # ExaAdmm.admm_outer_prestep(env2, mod2) #? not needed in one_level
    ExaAdmm.admm_increment_reset_inner(env2, mod2)
    ExaAdmm.admm_increment_inner(env2, mod2)
    # ExaAdmm.admm_inner_prestep(env2, mod2) #? not needed in one_level


    println("update x starts (no branch kernel)") 
    ExaAdmm.admm_update_x(env2, mod2)
    println(round.(sol2.u_curr,digits = 7))
    # println(mod2.sqp_line)
    # U_SOL_CPU = [-0.229424, -0.1339605, -0.0813327, -0.0049391, -0.0465958, 0.2252203, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # U_SOL_CPU2 = [-0.229424, -0.1339605, -0.0813327, -0.0049391, -0.0465958, 0.2252203, -0.1236772, -0.1271249, 0.1236772, 0.1189746, -0.1182455, -0.1040701, -0.0, 0.0022044, -0.0630247, -0.1092983, 0.0622891, 0.1082608, -0.0297814, -0.0074735, 0.0422504, 0.0451598, 0.0984247, 0.0700557, -0.102193, -0.0905333, 0.0294319, -0.0067947, 0.0250279, 0.0125998, -0.1368894, 0.2542204, 0.1368894, -0.2698603, -0.0770438, -0.1077549, -0.0375397, -0.0344878, -0.0665263, -0.1146067, 0.0658612, 0.1071325, -0.0032826, 0.0208988, -0.0055371, -0.0008479, 0.1049601, 0.1480269, -0.1061072, -0.1593811, 0.0230133, -0.0010432, -0.0026878, -0.0082759, 0.2045771, -0.05922, -0.2045771, 0.0340703, -0.0553384, -0.0495078, -0.0467405, -0.0542589, -0.1322638, -0.1856806, 0.1264451, 0.1533923, -0.0223818, 0.0420754, 0.0141644, 0.0280826, 0.0937455, 0.2743344, -0.0957246, -0.2938647, 0.0406687, -0.0099012, 0.0507601, 0.0458481]
    # println("x update error = ", norm(U_SOL_CPU2 - Array(sol2.u_curr)))
    # @test norm(U_SOL_CPU2 - Array(sol2.u_curr),  Inf) <= atol 

    println("update x_bar starts")
    ExaAdmm.admm_update_xbar(env2, mod2)
    println(round.(sol2.v_curr,digits = 7))
    # V_SOL_CPU = [-0.114712, -0.0669803, -0.0406664, -0.0024695, -0.0232979, 0.1126102, -0.114712, -0.0669803, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0232979, 0.1126102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0406664, -0.0024695, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # # println("x_bar update error = ", norm(V_SOL_CPU - Array(sol2.v_curr)))
    # # @test norm(V_SOL_CPU - Array(sol2.v_curr),  Inf) <= atol 
    
    
    println("update λ starts")
    ExaAdmm.admm_update_l_single(env2, mod2)
    println(round.(sol2.l_curr,digits = 7))
    # L_SOL_CPU = [-2.2942402, -1.3396053, -0.8133275, -0.0493907, -0.4659581, 2.2522035, 2.2942402, 1.3396053, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4659581, -2.2522035, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8133275, 0.0493907, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # # println("l update error = ", norm(L_SOL_CPU - Array(sol2.l_curr)))
    # # @test norm(L_SOL_CPU - Array(sol2.l_curr),  Inf) <= atol 



    println("update residual starts")
    ExaAdmm.admm_update_residual(env2, mod2)
    println(round.(sol2.rd,digits = 7))
    println(round.(sol2.rp,digits = 7))
    # RD_SOL_CPU = [-11.4745597, 1.3396053, -4.30396, 0.0493907, -9.0989271, -2.2522035, -2.2942402, 68.1048402, 0.0, 64.7764622, 4.0000001, 3.7311242, 0.0, -0.8053754, 7.2465568, 38.9219016, 6.415593, 34.4587212, 3.7311242, 3.3032756, -0.8053754, -1.2924705, 4.2348296, 17.8682271, 5.1280368, 21.636981, 3.3032756, 4.0000002, -1.2924705, 0.3894044, -0.4659581, 60.5216577, 0.0, 68.2593893, 3.41459, 4.0000002, 1.3392566, 0.3894044, 4.6203502, 38.7190838, 4.0887017, 34.2638063, 4.0000002, 3.5397334, 0.3894044, -0.2361645, 5.7241824, 48.2234817, 6.4684902, 54.4939173, 3.5397334, 4.0000002, -0.2361645, 0.4992945, 0.0, 64.0000034, -0.8133275, 62.1847399, 4.0000002, 3.8896332, 0.4992945, 1.9246906, 4.7504178, 23.2885394, 3.2690865, 16.0264325, 4.0000002, 2.752673, 0.4992945, -1.5132208, 3.7579153, 31.7000452, 5.0936849, 42.9679831, 2.752673, 3.7311242, -1.5132208, -0.8053754]
    # RP_SOL_CPU = [-0.114712, -0.0669803, -0.0406664, -0.0024695, -0.0232979, 0.1126102, 0.114712, 0.0669803, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0232979, -0.1126102, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0406664, 0.0024695, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # # println("dual residual error = ", norm(RD_SOL_CPU - Array(sol2.rd)))
    # # println("primal residual error = ", norm(RP_SOL_CPU - Array(sol2.rp)))
    # # @test norm(RD_SOL_CPU - Array(sol2.rd),  Inf) <= atol 
    # # @test norm(RP_SOL_CPU - Array(sol2.rp),  Inf) <= atol 
# end