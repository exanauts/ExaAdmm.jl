"""
    eval_linfeas_err()

- evaluate linear constraint violation given mod.sqp_sol
"""

function eval_linfeas_err(
    modsqp::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
    )
    data = modsqp.grid_data

    #generation bound constraint 
    err_pg_ub = [maximum([0,modsqp.pg_sol[g] - data.pgmax[g]]) for g = 1: data.ngen] #ngen
    err_pg_lb = [maximum([0, data.pgmin[g] - modsqp.pg_sol[g]]) for g = 1: data.ngen] #ngen
    err_qg_ub = [maximum([0,modsqp.qg_sol[g] - data.qgmax[g]]) for g = 1: data.ngen] #ngen
    err_qg_lb = [maximum([0, data.qgmin[g] - modsqp.qg_sol[g]]) for g = 1: data.ngen] #ngen

    #line flow upper bound constraint 
    err_pij_ub = [maximum([0,modsqp.line_fl[1,l] - sqrt(data.rateA[l])]) for l = 1: data.nline] #nline
    err_qij_ub = [maximum([0,modsqp.line_fl[2,l] - sqrt(data.rateA[l])]) for l = 1: data.nline] #nline
    err_pji_ub = [maximum([0,modsqp.line_fl[3,l] - sqrt(data.rateA[l])]) for l = 1: data.nline] #nline
    err_qji_ub = [maximum([0,modsqp.line_fl[4,l] - sqrt(data.rateA[l])]) for l = 1: data.nline]  #nline

    #line flow lower bound constraint 
    err_pij_lb = [maximum([0, -sqrt(data.rateA[l]) - modsqp.line_fl[1,l]]) for l = 1: data.nline] #nline
    err_qij_lb = [maximum([0, -sqrt(data.rateA[l]) - modsqp.line_fl[2,l]]) for l = 1: data.nline] #nline
    err_pji_lb = [maximum([0, -sqrt(data.rateA[l]) - modsqp.line_fl[3,l]]) for l = 1: data.nline] #nline
    err_qji_lb = [maximum([0, -sqrt(data.rateA[l]) - modsqp.line_fl[4,l]]) for l = 1: data.nline] #nline

    #line_var upper bound constraint 
    err_wijR_ub = [maximum([0,modsqp.line_var[1,l] - data.FrVmBound[2*l] * data.ToVmBound[2*l]]) for l = 1: data.nline] #nline
    err_wijI_ub = [maximum([0,modsqp.line_var[2,l] - data.FrVmBound[2*l] * data.ToVmBound[2*l]]) for l = 1: data.nline] #nline
    err_wi_ub = [maximum([0,modsqp.line_var[3,l] - data.FrVmBound[2*l]^2]) for l = 1: data.nline] #nline
    err_wj_ub = [maximum([0,modsqp.line_var[4,l] - data.ToVmBound[2*l]^2]) for l = 1: data.nline] #nline
    err_ti_ub = [maximum([0,modsqp.line_var[5,l] - data.FrVaBound[2*l]]) for l = 1: data.nline] #nline
    err_tj_ub = [maximum([0,modsqp.line_var[6,l] - data.ToVaBound[2*l]]) for l = 1: data.nline] #nline

    #line_var lower bound constraint 
    err_wijR_lb = [maximum([0,-modsqp.line_var[1,l] - data.FrVmBound[2*l] * data.ToVmBound[2*l]]) for l = 1: data.nline] #nline
    err_wijI_lb = [maximum([0,-modsqp.line_var[2,l] - data.FrVmBound[2*l] * data.ToVmBound[2*l]]) for l = 1: data.nline] #nline
    err_wi_lb = [maximum([0,-modsqp.line_var[3,l] + data.FrVmBound[2*l-1]^2]) for l = 1: data.nline] #nline
    err_wj_lb = [maximum([0,-modsqp.line_var[4,l] + data.ToVmBound[2*l-1]^2]) for l = 1: data.nline] #nline
    err_ti_lb = [maximum([0,-modsqp.line_var[5,l] + data.FrVaBound[2*l-1]]) for l = 1: data.nline] #nline
    err_tj_lb = [maximum([0,-modsqp.line_var[6,l] + data.ToVaBound[2*l-1]]) for l = 1: data.nline] #nline

    #match line_fl with line_var
    err_fl_var = zeros(data.nline)
    for l = 1: data.nline
        supY = [data.YftR[l] data.YftI[l] data.YffR[l] 0 0 0;
                -data.YftI[l] data.YftR[l] -data.YffI[l] 0 0 0;
                data.YtfR[l] -data.YtfI[l] 0 data.YttR[l] 0 0;
                -data.YtfI[l] -data.YtfR[l] 0 -data.YttI[l] 0 0]
        err_fl_var[l] = sum(abs.( supY * modsqp.line_var[:,l] - modsqp.line_fl[:,l]))
    end

    #consensus
    cons_errb = zeros(data.nbus)
    cons_errt = zeros(data.nbus)

    for b = 2:2 #1:data.nbus
        if data.FrStart[b] < data.FrStart[b+1] 
            for k = data.FrStart[b]:data.FrStart[b+1]-1
                cons_errb[b] += abs( modsqp.w_sol[b] - modsqp.line_var[3, data.FrIdx[k]]) #wi(ij)
                cons_errt[b] += abs(modsqp.theta_sol[b] - modsqp.line_var[5, data.FrIdx[k]]) #ti(ij)
                # print(cons_errb[b], " at 1")
            end
        end
        if data.ToStart[b] < data.ToStart[b+1]
            for k = data.ToStart[b]:data.ToStart[b+1]-1
                cons_errb[b] += abs( modsqp.w_sol[b] - modsqp.line_var[4, data.ToIdx[k]]) #wj(ji)
                cons_errt[b] += abs(modsqp.theta_sol[b] - modsqp.line_var[6, data.ToIdx[k]])#tj(ji)
                # print(modsqp.w_sol[b] )
                # print(modsqp.line_var[4, data.FrIdx[k]])
                # print(cons_errb[b], " at 2")
            end
        end
    end

    #power balance 
    #Pd 
    err_pd = zeros(data.nbus)
    pft = zeros(data.nbus)
    ptf = zeros(data.nbus)
    pgb = zeros(data.nbus)
    for b = 1:data.nbus
        if data.FrStart[b] < data.FrStart[b+1]
            pft[b] = sum( modsqp.line_fl[1,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1)
        end

        if data.ToStart[b] < data.ToStart[b+1]
            ptf[b] = sum( modsqp.line_fl[3,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1)
        end

        if data.GenStart[b] < data.GenStart[b+1]
            pgb[b] = sum( modsqp.pg_sol[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1)
        end

        err_pd[b] = abs( pgb[b] - pft[b] - ptf[b] - data.YshR[b]*modsqp.w_sol[b] - data.Pd[b]/data.baseMVA) 
    end

    #Qd 
    err_qd = zeros(data.nbus)
    qft = zeros(data.nbus)
    qtf = zeros(data.nbus)
    qgb = zeros(data.nbus)
    for b = 1:data.nbus
        if data.FrStart[b] < data.FrStart[b+1]
            qft[b] = sum( modsqp.line_fl[2,data.FrIdx[k]] for k = data.FrStart[b]:data.FrStart[b+1]-1)
        end

        if data.ToStart[b] < data.ToStart[b+1]
            qtf[b] = sum( modsqp.line_fl[4,data.ToIdx[k]] for k = data.ToStart[b]:data.ToStart[b+1]-1)
        end

        if data.GenStart[b] < data.GenStart[b+1]
            qgb[b] =sum( modsqp.qg_sol[data.GenIdx[g]] for g = data.GenStart[b]:data.GenStart[b+1]-1)
        end

        err_qd[b] = abs( qgb[b] - qft[b] - qtf[b] + data.YshI[b]*modsqp.w_sol[b] - data.Qd[b]/data.baseMVA) 
    end

    #for debug 
    # println(sum(err_pg_lb) + sum(err_pg_ub) + sum(err_qg_lb) + sum(err_qg_ub) )
    # println(sum(err_pij_ub) + sum(err_qij_ub) + sum(err_pji_ub) + sum(err_qji_ub))
    # println(sum(err_pij_lb) + sum(err_qij_lb) + sum(err_pji_lb) + sum(err_qji_lb))
    # println(sum(err_wijR_ub) + sum(err_wijI_ub) + sum(err_wi_ub) + sum(err_wj_ub) + sum(err_ti_ub) + sum(err_tj_ub) )
    # println(sum(err_wijR_lb) + sum(err_wijI_lb) + sum(err_wi_lb) + sum(err_wj_lb) + sum(err_ti_lb) + sum(err_tj_lb) )
    # println(sum(err_fl_var) )
    # println(sum(err_pd)) 
    # println(sum(err_qd) )
    # println(cons_errb)
    # println(sum(cons_err2))

    return sum(err_pg_lb) + sum(err_pg_ub) + sum(err_qg_lb) + sum(err_qg_ub) +
            sum(err_pij_ub) + sum(err_qij_ub) + sum(err_pji_ub) + sum(err_qji_ub) +
            sum(err_pij_lb) + sum(err_qij_lb) + sum(err_pji_lb) + sum(err_qji_lb) +
            sum(err_wijR_ub) + sum(err_wijI_ub) + sum(err_wi_ub) + sum(err_wj_ub) + sum(err_ti_ub) + sum(err_tj_ub) +
            sum(err_wijR_lb) + sum(err_wijI_lb) + sum(err_wi_lb) + sum(err_wj_lb) + sum(err_ti_lb) + sum(err_tj_lb) + 
            sum(err_fl_var) + sum(cons_errb) + sum(cons_errt) + sum(err_pd) + sum(err_qd)
end

"""
    lin_feas()

- solve linear feasibility problem given mod.sqp_sol
"""

function lin_feas(
    modsqp::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
    )
    data = modsqp.grid_data

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
    
        @variable(model, bus_w[b = 1:data.nbus], start = 1.0) #for consensus
        @variable(model, bus_theta[b = 1:data.nbus], start = 0.0) #for consensus 
    
    
    
    
        # objective: ||x-x_k||_2^2
        @objective(model, Min, sum( (pg[i] - modsqp.pg_sol[i])^2 + (qg[i] - modsqp.qg_sol[i])^2 for i = 1:data.ngen) 
                    + sum( (bus_w[i] - modsqp.w_sol[i])^2 + (bus_theta[i] - modsqp.theta_sol[i])^2 for i = 1:data.nbus) 
                    + sum( (line_var[i,j] - modsqp.line_var[i,j])^2 for i = 1:6, j = 1:data.nline) 
                    + sum( (line_fl[i,j] - modsqp.line_fl[i,j])^2 for i = 1:4, j = 1:data.nline) 
        )
    
    
        # generator bound constraint
        @constraint(model, [g=1:data.ngen], pg[g] <= data.pgmax[g])
        @constraint(model, [g=1:data.ngen], qg[g] <= data.qgmax[g])
        @constraint(model, [g=1:data.ngen], data.pgmin[g] <= pg[g] )
        @constraint(model, [g=1:data.ngen], data.qgmin[g] <= qg[g] )

       


        #line flow bound constraint 
        @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[1,l] <= sqrt(data.rateA[l])) #pij
        @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[2,l] <= sqrt(data.rateA[l])) #qij
        @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[3,l] <= sqrt(data.rateA[l])) #pji
        @constraint(model, [l=1:data.nline],  -sqrt(data.rateA[l]) <= line_fl[4,l] <= sqrt(data.rateA[l])) #qji
    
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
            @constraint(model, line_var[1,l] >= -data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijR
            @constraint(model, line_var[1,l] <= data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijR

            @constraint(model, line_var[2,l] >= -data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijI
            @constraint(model, line_var[2,l] <= data.FrVmBound[2*l] * data.ToVmBound[2*l]) #wijI

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
            if data.FrStart[b] < data.FrStart[b+1]
                for k = data.FrStart[b]:data.FrStart[b+1]-1
                    @constraint(model, bus_w[b] == line_var[3, data.FrIdx[k]]) #wi(ij)
                    @constraint(model, bus_theta[b] == line_var[5, data.FrIdx[k]]) #ti(ij)
                end
            end
            if data.ToStart[b] < data.ToStart[b+1]
                for k = data.ToStart[b]:data.ToStart[b+1]-1
                    @constraint(model, bus_w[b] == line_var[4, data.ToIdx[k]]) #wj(ji)
                    @constraint(model, bus_theta[b] == line_var[6, data.ToIdx[k]]) #tj(ji)
                end
            end
        end
    
    
        optimize!(model)
    
        println(termination_status(model)) #for debug 
    
    end #@inbounds

    #write solution into sqp model
    modsqp.pg_sol .= value.(pg)
    modsqp.qg_sol .= value.(qg)
    modsqp.line_var .= value.(line_var)
    modsqp.line_fl .= value.(line_fl)
    modsqp.w_sol .= value.(bus_w)
    modsqp.theta_sol .= value.(bus_theta)

    return
end