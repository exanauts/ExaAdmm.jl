"""
    sqp_qpsub_SOC()

- construct SOC modified from qpsub
- solve SOC

"""

function sqp_qpsub_SOC(
    mod1::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
    )

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

    return
end