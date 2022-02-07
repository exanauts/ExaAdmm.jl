function check_generator_bounds(model::Model, v::Vector{Float64})
    ngen = model.ngen
    gen_start = model.gen_start

    pgmax = model.pgmax_curr; pgmin = model.pgmin_curr
    qgmax = model.qgmax; qgmin = model.qgmin

    max_viol_real = 0.0
    max_viol_reactive = 0.0

    for g=1:ngen
        pidx = gen_start + 2*(g-1)
        qidx = gen_start + 2*(g-1) + 1

        real_err = max(max(0.0, v[pidx] - pgmax[g]), max(0.0, pgmin[g] - v[pidx]))
        reactive_err = max(max(0.0, v[qidx] - qgmax[g]), max(0.0, qgmin[g] - v[qidx]))

        max_viol_real = (max_viol_real < real_err) ? real_err : max_viol_real
        max_viol_reactive = (max_viol_reactive < reactive_err) ? reactive_err : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_voltage_bounds_alternative(model::Model, v::Vector{Float64})
    max_viol = 0.0

    for b=1:model.nbus
        if model.FrStart[b] < model.FrStart[b+1]
            l = model.FrIdx[model.FrStart[b]]
            wi = v[model.line_start + 8*(l-1) + 4]
        elseif model.ToStart[b] < model.ToStart[b+1]
            l = model.ToIdx[model.ToStart[b]]
            wi = v[model.line_start + 8*(l-1) + 5]
        else
            println("No lines connected to bus ", b)
        end

        err = max(max(0.0, wi - model.Vmax[b]^2), max(0.0, model.Vmin[b]^2 - wi))
        max_viol = (max_viol < err) ? err : max_viol
    end

    return max_viol
end

function check_power_balance_alternative(model::Model, u::Vector{Float64}, v::Vector{Float64})
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start = model.gen_start, model.line_start

    YffR, YffI = model.YffR, model.YffI
    YftR, YftI = model.YftR, model.YftI
    YtfR, YtfI = model.YtfR, model.YtfI
    YttR, YttI = model.YttR, model.YttI
    YshR, YshI = model.YshR, model.YshI

    vm = [0.0 for i=1:nbus]
    va = [0.0 for i=1:nbus]

    for b=1:nbus
        count = 0
        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            vm[b] = sqrt(v[line_start + 8*(l-1) + 4])
            va[b] = v[line_start + 8*(l-1) + 6]
            count += 1
        end
        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            vm[b] = sqrt(v[line_start + 8*(l-1) + 5])
            va[b] = v[line_start + 8*(l-1) + 7]
            count += 1
        end
        #vm[b] /= count
        #va[b] /= count
    end

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    max_viol_reactive_idx = -1
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for k=model.GenStart[b]:model.GenStart[b+1]-1
            g = model.GenIdx[k]
            real_err += v[gen_start + 2*(g-1)]
            reactive_err += v[gen_start + 2*(g-1)+1]
        end

        real_err -= (model.Pd[b] / baseMVA)
        reactive_err -= (model.Qd[b] / baseMVA)

        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            i = model.brBusIdx[1+2*(l-1)]
            j = model.brBusIdx[1+2*(l-1)+1]
            pij = YffR[l]*vm[i]^2 + YftR[l]*vm[i]*vm[j]*cos(va[i] - va[j]) + YftI[l]*vm[i]*vm[j]*sin(va[i] - va[j])
            qij = -YffI[l]*vm[i]^2 - YftI[l]*vm[i]*vm[j]*cos(va[i] - va[j]) + YftR[l]*vm[i]*vm[j]*sin(va[i] - va[j])
            real_err -= pij
            reactive_err -= qij
        end

        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            i = model.brBusIdx[1+2*(l-1)]
            j = model.brBusIdx[1+2*(l-1)+1]
            pji = YttR[l]*vm[j]^2 + YtfR[l]*vm[i]*vm[j]*cos(va[i] - va[j]) - YtfI[l]*vm[i]*vm[j]*sin(va[i] - va[j])
            qji = -YttI[l]*vm[j]^2 - YtfI[l]*vm[i]*vm[j]*cos(va[i] - va[j]) - YtfR[l]*vm[i]*vm[j]*sin(va[i] - va[j])
            real_err -= pji
            reactive_err -= qji
        end

        real_err -= YshR[b] * vm[b]^2
        reactive_err += YshI[b] * vm[b]^2

        if max_viol_reactive < abs(reactive_err)
            max_viol_reactive_idx = b
        end
        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_linelimit_violation(model::Model, data::OPFData, v::Vector{Float64})
    lines = data.lines
    nbus = length(data.buses)
    nline = length(data.lines)
    line_start = 2*length(data.generators) + 1

    YffR, YffI = model.YffR, model.YffI
    YftR, YftI = model.YftR, model.YftI
    YtfR, YtfI = model.YtfR, model.YtfI
    YttR, YttI = model.YttR, model.YttI

    rateA_nviols = 0
    rateA_maxviol = 0.0

    vm = [0.0 for i=1:nbus]
    va = [0.0 for i=1:nbus]

    for b=1:nbus
        count = 0
        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            vm[b] = sqrt(v[line_start + 8*(l-1) + 4])
            va[b] = v[line_start + 8*(l-1) + 6]
            count += 1
        end
        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            vm[b] = sqrt(v[line_start + 8*(l-1) + 5])
            va[b] = v[line_start + 8*(l-1) + 7]
            count += 1
        end
        #vm[b] /= count
        #va[b] /= count
    end

    for l=1:nline
        i = model.brBusIdx[1+2*(l-1)]
        j = model.brBusIdx[1+2*(l-1)+1]
        pij = YffR[l]*vm[i]^2 + YftR[l]*vm[i]*vm[j]*cos(va[i] - va[j]) + YftI[l]*vm[i]*vm[j]*sin(va[i] - va[j])
        qij = -YffI[l]*vm[i]^2 - YftI[l]*vm[i]*vm[j]*cos(va[i] - va[j]) + YftR[l]*vm[i]*vm[j]*sin(va[i] - va[j])
        pji = YttR[l]*vm[j]^2 + YtfR[l]*vm[i]*vm[j]*cos(va[i] - va[j]) - YtfI[l]*vm[i]*vm[j]*sin(va[i] - va[j])
        qji = -YttI[l]*vm[j]^2 - YtfI[l]*vm[i]*vm[j]*cos(va[i] - va[j]) - YtfR[l]*vm[i]*vm[j]*sin(va[i] - va[j])

        ij_val = sqrt(pij^2 + qij^2)
        ji_val = sqrt(pji^2 + qji^2)

        limit = lines[l].rateA / data.baseMVA
        if limit > 0
            if ij_val > limit || ji_val > limit
                rateA_nviols += 1
                rateA_maxviol = max(rateA_maxviol, max(ij_val - limit, ji_val - limit))
            end
        end
    end

    return rateA_nviols, rateA_maxviol
end

#=
function check_power_balance_alternative(model::Model, u::Vector{Float64}, v::Vector{Float64})
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start, YshR, YshI = model.gen_start, model.line_start, model.YshR, model.YshI

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        real_err = 0.0
        reactive_err = 0.0
        for k=model.GenStart[b]:model.GenStart[b+1]-1
            g = model.GenIdx[k]
            real_err += u[gen_start + 2*(g-1)]
            reactive_err += u[gen_start + 2*(g-1)+1]
        end

        real_err -= (model.Pd[b] / baseMVA)
        reactive_err -= (model.Qd[b] / baseMVA)

        wi = 0
        for k=model.FrStart[b]:model.FrStart[b+1]-1
            l = model.FrIdx[k]
            real_err -= v[line_start + 8*(l-1)]
            reactive_err -= v[line_start + 8*(l-1) + 1]
            wi = v[line_start + 8*(l-1) + 4]
        end

        for k=model.ToStart[b]:model.ToStart[b+1]-1
            l = model.ToIdx[k]
            real_err -= v[line_start + 8*(l-1) + 2]
            reactive_err -= v[line_start + 8*(l-1) + 3]
            wi = v[line_start + 8*(l-1) + 5]
        end

        real_err -= YshR[b] * wi
        reactive_err += YshI[b] * wi

        max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
        max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
    end

    return max_viol_real, max_viol_reactive
end

function check_linelimit_violation(data::OPFData, v::Vector{Float64})
    lines = data.lines
    nline = length(data.lines)
    line_start = 2*length(data.generators) + 1

    rateA_nviols = 0
    rateA_maxviol = 0.0

    for l=1:nline
        pij_idx = line_start + 8*(l-1)
        ij_val = sqrt(v[pij_idx]^2 + v[pij_idx+1]^2)
        ji_val = sqrt(v[pij_idx+2]^2 + v[pij_idx+3]^2)

        limit = lines[l].rateA / data.baseMVA
        if limit > 0
            if ij_val > limit || ji_val > limit
                rateA_nviols += 1
                rateA_maxviol = max(rateA_maxviol, max(ij_val - limit, ji_val - limit))
            end
        end
    end

    return rateA_nviols, rateA_maxviol
end
=#