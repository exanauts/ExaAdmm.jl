
    
case = joinpath(INSTANCES_DIR, "case9.m")
rho_pq = 20.0 #for two level 
rho_va = 20.0 #for two level
initial_beta = 100000.0 #for two level
verbose = 0
fix_line = false
    
    
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
    @objective(model2, Min, sum(mod1.qpsub_c2[g]*(pg[g]*data.baseMVA)^2 + mod1.qpsub_c1[g]*pg[g]*data.baseMVA for g=1:data.ngen) +
        sum(0.5*dot(line_var[:,l],mod1.Hs[6*(l-1)+1:6*l,1:6],line_var[:,l]) for l=1:data.nline) )
    
    
    
    
    # generator constraint
    @constraint(model2, [g=1:data.ngen], pg[g] <= mod1.qpsub_pgmax[g])
    @constraint(model2, [g=1:data.ngen], qg[g] <= mod1.qpsub_qgmax[g])
    @constraint(model2, [g=1:data.ngen], mod1.qpsub_pgmin[g] <= pg[g] )
    @constraint(model2, [g=1:data.ngen], mod1.qpsub_qgmin[g] <= qg[g] )
    
    
    
    
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
    
        @constraint(model2, pgb[b] - pft[b] - ptf[b] - data.YshR[b]*bus_w[b] == mod1.qpsub_Pd[b]/data.baseMVA) 
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
    
        @constraint(model2, qgb[b] - qft[b] - qtf[b] + data.YshI[b]*bus_w[b] == mod1.qpsub_Qd[b]/data.baseMVA) 
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

# test one-iteration update    
@testset "Testing [x,xbar,l,residual] updates" begin     
    atol = 2e-6
    env2 = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
    mod2 = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env2)
    sol2 = mod2.solution
    par2 = env2.params
    data2 = mod2.grid_data
    info2 = mod2.info
    
    env2.params.scale = 1e-4
    env2.params.initial_beta = 1e3
    env2.params.beta = 1e3
    
    mod2.Hs .= mod1.Hs
    mod2.LH_1h .= mod1.LH_1h
    mod2.RH_1h .= mod1.RH_1h
    mod2.LH_1i .= mod1.LH_1i
    mod2.RH_1i .= mod1.RH_1i
    mod2.LH_1j .= mod1.LH_1j
    mod2.RH_1j .= mod1.RH_1j
    mod2.LH_1k .= mod1.LH_1k
    mod2.RH_1k .= mod1.RH_1k
    mod2.ls .= mod1.ls
    mod2.us .= mod1.us
    
    mod2.qpsub_pgmax .= mod1.qpsub_pgmax
    mod2.qpsub_pgmin .= mod1.qpsub_pgmin
    mod2.qpsub_qgmax .= mod1.qpsub_qgmax
    mod2.qpsub_qgmin .= mod1.qpsub_qgmin

    mod2.qpsub_c1 .= mod1.qpsub_c1
    mod2.qpsub_c2 .= mod1.qpsub_c2
    mod2.qpsub_Pd .= mod1.qpsub_Pd
    mod2.qpsub_Qd .= mod1.qpsub_Qd
    
    ExaAdmm.init_solution!(mod2, mod2.solution, env2.initial_rho_pq, env2.initial_rho_va)
    
    info2.norm_z_prev = info2.norm_z_curr = 0
    par2.initial_beta = 0 
    par2.beta = 0
    sol2.lz .= 0
    sol2.z_curr .= 0
    sol2.z_prev .= 0
    par2.inner_iterlim = 1
    
    println("prestep starts")
    ExaAdmm.admm_increment_outer(env2, mod2)
    ExaAdmm.admm_outer_prestep(env2, mod2)
    ExaAdmm.admm_increment_reset_inner(env2, mod2)
    ExaAdmm.admm_increment_inner(env2, mod2)
    ExaAdmm.admm_inner_prestep(env2, mod2)
    
    println("update x starts")
    ExaAdmm.admm_update_x(env2, mod2)
    U_SOL = [-0.229424, -0.1339605, -0.0813327, -0.0049391, -0.0465958, 0.2252203, -0.1236772, -0.1271249, 0.1236772, 0.1189746, -0.1182455, -0.1040701, -0.0, 0.0022044, -0.0630247, -0.1092983, 0.0622891, 0.1082608, -0.0297814, -0.0074735, 0.0422504, 0.0451598, 0.0984247, 0.0700557, -0.102193, -0.0905333, 0.0294319, -0.0067947, 0.0250279, 0.0125998, -0.1368894, 0.2542204, 0.1368894, -0.2698603, -0.0770438, -0.1077549, -0.0375397, -0.0344878, -0.0665263, -0.1146067, 0.0658612, 0.1071325, -0.0032826, 0.0208988, -0.0055371, -0.0008479, 0.1049601, 0.1480269, -0.1061072, -0.1593811, 0.0230133, -0.0010432, -0.0026878, -0.0082759, 0.2045771, -0.05922, -0.2045771, 0.0340703, -0.0553384, -0.0495078, -0.0467405, -0.0542589, -0.1322638, -0.1856806, 0.1264451, 0.1533923, -0.0223818, 0.0420754, 0.0141644, 0.0280826, 0.0937455, 0.2743344, -0.0957246, -0.2938647, 0.0406687, -0.0099012, 0.0507601, 0.0458481]
    @test norm( sol2.u_curr .- U_SOL, Inf) <= atol
    
    
    println("update x_bar starts")
    ExaAdmm.admm_update_xbar(env2, mod2)
    V_SOL = [-0.1765506, -0.1305427, -0.1429549, 0.0145656, -0.0917426, 0.2397204, -0.1765506, -0.1305427, 0.1353679, 0.2137041, -0.1182455, -0.0479176, -0.0, 0.030101, -0.051334, -0.0145688, -0.0180678, 0.0191025, -0.0479176, 0.0109792, 0.030101, 0.0350939, 0.0180678, -0.0191025, -0.091583, 0.0678001, 0.0109792, -0.0392774, 0.0350939, -0.0091417, -0.0917426, 0.2397204, 0.1474993, -0.1115269, -0.0770438, -0.0392774, -0.0375397, -0.0091417, -0.0559163, 0.0437267, -0.0195494, -0.0204472, -0.0392774, 0.0219561, -0.0091417, -0.0017678, 0.0195494, 0.0204472, -0.0948426, -0.0246205, 0.0219561, -0.0262545, -0.0017678, -0.0136173, 0.2158417, 0.0755405, -0.1429549, 0.0145656, -0.0262545, -0.0495078, -0.0136173, -0.0542589, -0.1209991, -0.05092, 0.0163498, -0.0604711, -0.0262545, 0.041372, -0.0136173, 0.0394213, -0.0163498, 0.0604711, -0.0840339, -0.1991353, 0.041372, -0.0479176, 0.0394213, 0.030101]
    @test norm( sol2.v_curr .- V_SOL, Inf) <= atol
    


    println("update λ starts")
    ExaAdmm.admm_update_l_single(env2, mod2)
    L_SOL = [-1.057468, -0.0683565, 1.2324431, -0.390094, 0.9029359, -0.2900002, 1.057468, 0.0683565, -0.2338139, -1.8945894, 0.0, -1.1230512, 0.0, -0.5579311, -0.2338139, -1.8945894, 1.6071386, 1.7831649, 0.362723, -0.3690544, 0.2429884, 0.201319, 1.6071386, 1.7831649, -0.2121989, -3.166669, 0.3690544, 0.6496544, -0.201319, 0.4348308, -0.9029359, 0.2900002, -0.2121989, -3.166669, 0.0, -1.3695503, 0.0, -0.5069225, -0.2121989, -3.166669, 1.7082129, 2.5515935, 0.7198958, -0.0211449, 0.0720918, 0.0183997, 1.7082129, 2.5515935, -0.2252933, -2.6952112, 0.0211449, 0.5042264, -0.0183997, 0.1068284, -0.2252933, -2.6952112, -1.2324431, 0.390094, -0.5816791, 0.0, -0.6624625, 0.0, -0.2252933, -2.6952112, 2.2019063, 4.277267, 0.0774527, 0.0140663, 0.5556341, -0.2267749, 2.2019063, 4.277267, -0.2338139, -1.8945894, -0.0140663, 0.7603282, 0.2267749, 0.3149427]
    @test norm( sol2.l_curr .- L_SOL, Inf) <= atol
    
    
    println("update residual starts")
    ExaAdmm.admm_update_residual(env2, mod2)
    RP_SOL = [-0.0528734, -0.0034178, 0.0616222, -0.0195047, 0.0451468, -0.0145, 0.0528734, 0.0034178, -0.0116907, -0.0947295, 0.0, -0.0561526, 0.0, -0.0278966, -0.0116907, -0.0947295, 0.0803569, 0.0891582, 0.0181362, -0.0184527, 0.0121494, 0.0100659, 0.0803569, 0.0891582, -0.0106099, -0.1583335, 0.0184527, 0.0324827, -0.0100659, 0.0217415, -0.0451468, 0.0145, -0.0106099, -0.1583335, 0.0, -0.0684775, 0.0, -0.0253461, -0.0106099, -0.1583335, 0.0854106, 0.1275797, 0.0359948, -0.0010572, 0.0036046, 0.00092, 0.0854106, 0.1275797, -0.0112647, -0.1347606, 0.0010572, 0.0252113, -0.00092, 0.0053414, -0.0112647, -0.1347606, -0.0616222, 0.0195047, -0.029084, 0.0, -0.0331231, 0.0, -0.0112647, -0.1347606, 0.1100953, 0.2138633, 0.0038726, 0.0007033, 0.0277817, -0.0113387, 0.1100953, 0.2138633, -0.0116907, -0.0947295, -0.0007033, 0.0380164, 0.0113387, 0.0157471]
    RD_SOL = [-12.7113319, 0.0683565, -6.3497307, 0.390094, -10.4678211, 0.2900002, -3.5310125, 66.8335931, 2.7073584, 69.0505462, 1.6350909, 2.7727729, 0.0, -0.2033556, 6.2198775, 38.6305254, 6.0542368, 34.8407725, 2.7727729, 3.5228599, -0.2033556, -0.5905927, 4.596186, 17.4861768, 3.2963767, 22.9929842, 3.5228599, 3.2144525, -0.5905927, 0.2065699, -1.8348521, 63.0638615, 2.9499869, 66.0288518, 1.8737146, 3.2144525, 0.5884628, 0.2065699, 3.5020235, 39.5936187, 3.6977135, 33.8548621, 3.2144525, 3.9788551, 0.2065699, -0.2715214, 6.1151706, 48.6324263, 4.5716387, 54.0015076, 3.9788551, 3.4749108, -0.2715214, 0.2269477, 4.3168345, 65.510814, -2.8590981, 62.5254434, 3.4749108, 2.8994775, 0.2269477, 0.8395123, 2.3304349, 22.2701391, 3.5960831, 14.8170115, 3.4749108, 3.580114, 0.2269477, -0.7247947, 3.4309189, 32.9094673, 3.4130062, 38.9852785, 3.580114, 2.7727729, -0.7247947, -0.2033556]

    @test norm( sol2.rp .- RP_SOL, Inf) <= atol
    @test norm( sol2.rd .- RD_SOL, Inf) <= atol

    
end #@testset

# test branch accuracy
@testset "branch ALM vs IPOPT" begin
    atol = 4e-5
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
        mod1.LH_1h[i,:], mod1.RH_1h[i], mod1.LH_1i[i,:], mod1.RH_1i[i], mod1.LH_1j[i,:], mod1.RH_1j[i], mod1.LH_1k[i,:], mod1.RH_1k[i], mod1.lambda)

        res[i] = norm(tronx[3:8] - x_ipopt1, Inf)
        res2[i] = norm(tronx[3:8] - x_ipopt2, Inf)
        res3[i] = norm(tronx[3:8] - tronx2[3:8], Inf)

        end #for
    end #inbound
    @test norm(res, Inf) <= atol
    @test norm(res2, Inf) <= atol
    @test norm(res3, Inf) <= atol
end # testset


#test solution 
@testset "Qpsub ADMM vs IPOPT" begin
    env3, mod3 = ExaAdmm.solve_qpsub(case, mod1.Hs, mod1.LH_1h, mod1.RH_1h,
    mod1.LH_1i, mod1.RH_1i, mod1.LH_1j, mod1.RH_1j, mod1.LH_1k, mod1.RH_1k, mod1.ls, mod1.us, mod1.qpsub_pgmax, mod1.qpsub_pgmin, mod1.qpsub_qgmax, mod1.qpsub_qgmin, mod1.qpsub_c1, mod1.qpsub_c2, mod1.qpsub_Pd, mod1.qpsub_Qd,
    initial_beta; 
        outer_iterlim=10000, inner_iterlim=1, scale = 1e-4, obj_scale = 1, rho_pq = 4000.0, rho_va = 4000.0, verbose=0, outer_eps=2*1e-6, onelevel = true)

  
    @test mod3.info.status == :Solved
    @test mod3.info.outer == 5107
    @test mod3.info.cumul == 5107
    @test isapprox(mod3.info.objval, objective_value(model2); atol=1e-2)
    @test isapprox(mod3.info.objval, -21.92744641968529; atol=1e-6)     
end 