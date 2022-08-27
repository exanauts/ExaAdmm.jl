# using Test
# using LazyArtifacts
# using LinearAlgebra
# using Printf


# using ExaAdmm
# using Random
# using JuMP
# using Ipopt


# INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
# MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "matpower")

@testset "Testing [x,xbar,l,res] updates and a solve for ACOPF" begin

rho_pq = 20.0 #for two level 
rho_va = 20.0 #for two level
initial_beta = 100000.0 #for two level
atol = 2e-6
verbose = 0
fix_line = false
use_gpu = true

case = joinpath(INSTANCES_DIR, "case9.m")

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


    println("update x starts (with branch kernel)") 
    ExaAdmm.admm_update_x(env2, mod2)
    # println(round.(sol2.u_curr,digits = 7))
    U_SOL_CPU = [-0.229424, -0.1339605, -0.0813327, -0.0049391, -0.0465958, 0.2252203, -0.1236772, -0.1271249, 0.1236772, 0.1189746, -0.1182455, -0.1040701, -0.0, 0.0022044, -0.0630247, -0.1092983, 0.0622891, 0.1082608, -0.0297814, -0.0074735, 0.0422504, 0.0451598, 0.0984247, 0.0700557, -0.102193, -0.0905333, 0.0294319, -0.0067947, 0.0250279, 0.0125998, -0.1368894, 0.2542204, 0.1368894, -0.2698603, -0.0770438, -0.1077549, -0.0375397, -0.0344878, -0.0665263, -0.1146067, 0.0658612, 0.1071325, -0.0032826, 0.0208988, -0.0055371, -0.0008479, 0.1049601, 0.1480269, -0.1061072, -0.1593811, 0.0230133, -0.0010432, -0.0026878, -0.0082759, 0.2045771, -0.05922, -0.2045771, 0.0340703, -0.0553384, -0.0495078, -0.0467405, -0.0542589, -0.1322638, -0.1856806, 0.1264451, 0.1533923, -0.0223818, 0.0420754, 0.0141644, 0.0280826, 0.0937455, 0.2743344, -0.0957246, -0.2938647, 0.0406687, -0.0099012, 0.0507601, 0.0458481]
    # println("x update error = ", norm(U_SOL_CPU - Array(sol2.u_curr)))
    @test norm(U_SOL_CPU - Array(sol2.u_curr),  Inf) <= atol 

    println("update x_bar starts")
    ExaAdmm.admm_update_xbar(env2, mod2)
    # println(round.(sol2.v_curr,digits = 7))
    V_SOL_CPU = [-0.1765506, -0.1305427, -0.1429549, 0.0145656, -0.0917426, 0.2397204, -0.1765506, -0.1305427, 0.1353679, 0.2137041, -0.1182455, -0.0479176, -0.0, 0.030101, -0.051334, -0.0145688, -0.0180678, 0.0191025, -0.0479176, 0.0109792, 0.030101, 0.0350939, 0.0180678, -0.0191025, -0.091583, 0.0678001, 0.0109792, -0.0392774, 0.0350939, -0.0091417, -0.0917426, 0.2397204, 0.1474993, -0.1115269, -0.0770438, -0.0392774, -0.0375397, -0.0091417, -0.0559163, 0.0437267, -0.0195494, -0.0204472, -0.0392774, 0.0219561, -0.0091417, -0.0017678, 0.0195494, 0.0204472, -0.0948426, -0.0246205, 0.0219561, -0.0262545, -0.0017678, -0.0136173, 0.2158417, 0.0755405, -0.1429549, 0.0145656, -0.0262545, -0.0495078, -0.0136173, -0.0542589, -0.1209991, -0.05092, 0.0163498, -0.0604711, -0.0262545, 0.041372, -0.0136173, 0.0394213, -0.0163498, 0.0604711, -0.0840339, -0.1991353, 0.041372, -0.0479176, 0.0394213, 0.030101]
    # println("x_bar update error = ", norm(V_SOL_CPU - Array(sol2.v_curr)))
    @test norm(V_SOL_CPU - Array(sol2.v_curr),  Inf) <= atol 
    
    
    println("update λ starts")
    ExaAdmm.admm_update_l_single(env2, mod2)
    # println(round.(sol2.l_curr,digits = 7))
    L_SOL_CPU = [-1.057468, -0.0683565, 1.2324431, -0.390094, 0.9029359, -0.2900002, 1.057468, 0.0683565, -0.2338139, -1.8945894, 0.0, -1.1230512, 0.0, -0.5579311, -0.2338139, -1.8945894, 1.6071386, 1.7831649, 0.362723, -0.3690544, 0.2429884, 0.201319, 1.6071386, 1.7831649, -0.2121989, -3.166669, 0.3690544, 0.6496544, -0.201319, 0.4348308, -0.9029359, 0.2900002, -0.2121989, -3.166669, 0.0, -1.3695503, 0.0, -0.5069225, -0.2121989, -3.166669, 1.7082129, 2.5515935, 0.7198958, -0.0211449, 0.0720918, 0.0183997, 1.7082129, 2.5515935, -0.2252933, -2.6952112, 0.0211449, 0.5042264, -0.0183997, 0.1068284, -0.2252933, -2.6952112, -1.2324431, 0.390094, -0.5816791, 0.0, -0.6624625, 0.0, -0.2252933, -2.6952112, 2.2019063, 4.277267, 0.0774527, 0.0140663, 0.5556341, -0.2267749, 2.2019063, 4.277267, -0.2338139, -1.8945894, -0.0140663, 0.7603282, 0.2267749, 0.3149427]
    # println("l update error = ", norm(L_SOL_CPU - Array(sol2.l_curr)))
    @test norm(L_SOL_CPU - Array(sol2.l_curr),  Inf) <= atol 



    println("update residual starts")
    ExaAdmm.admm_update_residual(env2, mod2)
    # println(round.(sol2.rd,digits = 7))
    # println(round.(sol2.rp,digits = 7))
    RD_SOL_CPU = [-12.7113319, 0.0683565, -6.3497307, 0.390094, -10.4678211, 0.2900002, -3.5310124, 66.8335914, 2.7073584, 69.0505445, 1.6350909, 2.7727729, 0.0, -0.2033556, 6.2198773, 38.6305246, 6.0542367, 34.8407717, 2.7727729, 3.5228599, -0.2033556, -0.5905927, 4.5961859, 17.4861766, 3.2963767, 22.992984, 3.5228599, 3.2144525, -0.5905927, 0.2065699, -1.8348521, 63.0638614, 2.9499869, 66.0288516, 1.8737146, 3.2144525, 0.5884628, 0.2065699, 3.5020235, 39.5936185, 3.6977135, 33.8548619, 3.2144525, 3.9788551, 0.2065699, -0.2715214, 6.1151706, 48.632426, 4.5716386, 54.0015074, 3.9788551, 3.4749108, -0.2715214, 0.2269477, 4.3168345, 65.5108139, -2.8590981, 62.5254432, 3.4749108, 2.8994775, 0.2269477, 0.8395123, 2.3304348, 22.2701389, 3.596083, 14.8170113, 3.4749108, 3.5801139, 0.2269477, -0.7247947, 3.4309188, 32.9094664, 3.4130061, 38.9852777, 3.5801139, 2.7727729, -0.7247947, -0.2033556]
    RP_SOL_CPU = [-0.0528734, -0.0034178, 0.0616222, -0.0195047, 0.0451468, -0.0145, 0.0528734, 0.0034178, -0.0116907, -0.0947295, 0.0, -0.0561526, 0.0, -0.0278966, -0.0116907, -0.0947295, 0.0803569, 0.0891582, 0.0181362, -0.0184527, 0.0121494, 0.0100659, 0.0803569, 0.0891582, -0.0106099, -0.1583335, 0.0184527, 0.0324827, -0.0100659, 0.0217415, -0.0451468, 0.0145, -0.0106099, -0.1583335, 0.0, -0.0684775, 0.0, -0.0253461, -0.0106099, -0.1583335, 0.0854106, 0.1275797, 0.0359948, -0.0010572, 0.0036046, 0.00092, 0.0854106, 0.1275797, -0.0112647, -0.1347606, 0.0010572, 0.0252113, -0.00092, 0.0053414, -0.0112647, -0.1347606, -0.0616222, 0.0195047, -0.029084, 0.0, -0.0331231, 0.0, -0.0112647, -0.1347606, 0.1100953, 0.2138633, 0.0038726, 0.0007033, 0.0277817, -0.0113387, 0.1100953, 0.2138633, -0.0116907, -0.0947295, -0.0007033, 0.0380164, 0.0113387, 0.0157471]
    # println("dual residual error = ", norm(RD_SOL_CPU - Array(sol2.rd)))
    # println("primal residual error = ", norm(RP_SOL_CPU - Array(sol2.rp)))
    @test norm(RD_SOL_CPU - Array(sol2.rd),  Inf) <= atol 
    @test norm(RP_SOL_CPU - Array(sol2.rp),  Inf) <= atol 
end


@testset "Qpsub ADMM (GPU) vs IPOPT vs ADMM (CPU)" begin
    env3, mod3 = ExaAdmm.solve_qpsub(case, mod1.Hs, mod1.LH_1h, mod1.RH_1h,
    mod1.LH_1i, mod1.RH_1i, mod1.LH_1j, mod1.RH_1j, mod1.LH_1k, mod1.RH_1k, mod1.ls, mod1.us, mod1.qpsub_pgmax, mod1.qpsub_pgmin, mod1.qpsub_qgmax, mod1.qpsub_qgmin, mod1.qpsub_c1, mod1.qpsub_c2, mod1.qpsub_Pd, mod1.qpsub_Qd,
    initial_beta; 
        outer_iterlim=10000, inner_iterlim=1, scale = 1e-4, obj_scale = 1, rho_pq = 4000.0, rho_va = 4000.0, verbose=0, outer_eps=2*1e-6, onelevel = true, use_gpu = true, gpu_no = 1)

  
    @test mod3.info.status == :Solved
    @test mod3.info.outer == 5107
    @test mod3.info.cumul == 5107
    @test isapprox(mod3.info.objval, objective_value(model2); atol=1e-2)
    @test isapprox(mod3.info.objval, -21.92744641968529; atol=1e-6)    
    
    sol3 = mod3.solution
    atol = 1e-6
# end 

# @testset "Qpsub ADMM (GPU) vs ADMM (CPU) Solution" begin
    dpg_sol_cpu = [-0.11550455128840183
    0.06549995591228526
    0.053676513010466456]

    dqg_sol_cpu = [-0.007903958357205233
    0.019785362536583685
    0.011746408694041054]

    dline_var_cpu = [-0.00630369   -0.00556137  -0.00333908   0.000404039   9.04974e-6   -8.98924e-6    0.000661247  -0.00373931  -0.00567469
    -0.00665306   -0.00509225  -0.00938981   0.00314544   -0.000313391  -0.000223849  -0.00409375    0.00970327   0.00512255
    -0.00675896   -0.0063758   -0.00500354   0.00109238   -1.06479e-8   -1.48199e-6   -1.06804e-8   -1.06804e-8  -0.00534789
    -0.00637581   -0.00500357  -1.06479e-8  -1.06479e-8   -1.48608e-6   -1.06804e-8    0.00189783   -0.00534788  -0.0063758
    -3.25492e-19   0.00531377   0.00949253   0.0201868     0.0175742     0.0178357     0.0180226     0.0180226    0.00951113
     0.00531381    0.00949261   0.0175743    0.0175742     0.0178358     0.0180226     0.021366      0.00951121   0.00531385]

     dline_fl_cpu = [ -0.115505    -0.0551048   -0.0546064    0.0536765   -0.00307677   -0.00305414   -0.0655      0.0624192    0.0598887
     -0.00790396   0.00183362   0.00363203   0.0117464    0.000273346   0.000464936  -0.0105801   0.0108191   -0.00273044
      0.115505     0.0546064    0.0567533   -0.0536765    0.00305414    0.0030808     0.0655     -0.0598887   -0.0603997
     -0.00125214  -0.00363203   0.00662172  -0.00689505  -0.000464924  -0.000239002   0.0197854   0.00273044  -0.000581486]

     dtheta_sol_cpu = [-3.25491739361945e-19
     0.021365998519708898
     0.020186769395364127
     0.005313812154732135
     0.009492570006911722
     0.017574220141571076
     0.017835751182986154
     0.01802260774972062
     0.009511171536802466]

     dw_sol_cpu = [-0.006758956828784941
     0.0018978329222831053
     0.0010923793242977361
    -0.006375802955645012
    -0.005003557142581212
    -1.0647877468628053e-8
    -1.484035364757708e-6
    -1.068039123808262e-8
    -0.0053478835040742636]

    dual_infeas_1_cpu = [-254.11001283448402
    111.34992505088496
    131.50745687564282
      0.31062229996245705
     -8.03424172781389
     -0.3078548536034032
     -0.07573915941618026
      0.012603427412195068
      0.008651826283358158
      0.03279612499623894] 

    dual_infeas_2_cpu = [-2.337593793742112
    -0.2062996739647522
     0.10994572761041102
     0.031770554831216935
     0.02745497289931917
    -0.11873053563714611
    -1.2564033339874354
    -0.12073294817351554
     0.2121498763329952
     0.05086202591219009]

    dual_infeas_3_cpu = [0.057405134084524315
    -0.16737701814667477
     3.670092553395319
     0.4052710265115234
    -0.2342336895471848
     0.07724866682808983
     0.07379529136930803
     0.004080775509334406
    -0.12229701235980252
    -0.0018175498192630844]

    dual_infeas_4_cpu = [-0.0019510899995607155
    0.07059765612753019
    0.07104222896868044
   -0.0061070561232362
   -0.17080362240460334
    0.0027843138173676155
    0.003450443341928419
    0.07157328724238488
    0.07186014778306046
   -0.28991023775276664]

    dual_infeas_5_cpu = [-4.200422692029046
    -0.33666635407314044
     0.6407231013041463
     0.0753794333813387
     0.08217499064640721
    -0.18387848500073856
     1.4609510208883514
     0.26889163927860954
    -0.13184568446925882
     0.065799538338272
     0.04433557788772256
     0.06271507036787678
     2.807937471979002
     0.0728632119983576
    -0.20915786828282304
     0.03875039932653859
     0.020549534704044203]

     dual_infeas_cpu = vcat(dual_infeas_1_cpu,dual_infeas_2_cpu,dual_infeas_3_cpu,dual_infeas_4_cpu,dual_infeas_5_cpu)

     @test norm(dpg_sol_cpu - Array(mod3.dpg_sol),  Inf) <= atol 
     @test norm(dqg_sol_cpu - Array(mod3.dqg_sol),  Inf) <= atol 
     @test norm(dline_var_cpu - Array(mod3.dline_var),  Inf) <= atol 
     @test norm(dline_fl_cpu - Array(mod3.dline_fl),  Inf) <= atol 
     @test norm(dtheta_sol_cpu - Array(mod3.dtheta_sol),  Inf) <= atol 
     @test norm(dw_sol_cpu - Array(mod3.dw_sol),  Inf) <= atol 
     @test norm(dual_infeas_cpu - Array(mod3.dual_infeas),  Inf) <= atol 
end