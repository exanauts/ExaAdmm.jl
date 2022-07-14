"""
    eval_nonlinfeas_err()

- evaluate nonlinear/all constraint violation given mod.sqp_sol
"""

function eval_nonlinfeas_err(
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
    eval_qpsubfeas_err()

- for qpsub, evaluate linear/all constraint violation given mod.qpsub_sol
"""

function eval_qpsubfeas_err(
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