function eval_f_real(pf::PowerFlow, x::Vector{Float64}, b::Int)
    nw = pf.nw
    baseMVA = nw["baseMVA"]

    f = 0.0
    i = pf.var_vmva_start + 2*(b-1)

    f += nw["YshR"][b]*x[i]^2
    for l in nw["frombus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["tbus"])]-1)
        f += nw["YffR"][l]*x[i]^2 + nw["YftR"][l]*x[i]*x[j]*cos(x[i+1]-x[j+1]) +
                                    nw["YftI"][l]*x[i]*x[j]*sin(x[i+1]-x[j+1])
    end

    for l in nw["tobus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["fbus"])]-1)
        f += nw["YttR"][l]*x[i]^2 + nw["YtfR"][l]*x[i]*x[j]*cos(x[i+1]-x[j+1]) +
                                    nw["YtfI"][l]*x[i]*x[j]*sin(x[i+1]-x[j+1])
    end

    gen = 0.0
    for g in nw["busgen"][b]
        j = pf.var_pgqg_start + 2*(g-1)
        gen += x[j]*baseMVA
    end

    f -= (gen-nw["bus"][b]["Pd"])/baseMVA
    return f
end

function eval_f_reactive(pf::PowerFlow, x::Vector{Float64}, b::Int)
    nw = pf.nw
    baseMVA = nw["baseMVA"]

    f = 0.0
    i = pf.var_vmva_start + 2*(b-1)

    f += (-nw["YshI"][b])*x[i]^2
    for l in nw["frombus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["tbus"])]-1)
        f += (-nw["YffI"][l])*x[i]^2 - nw["YftI"][l]*x[i]*x[j]*cos(x[i+1]-x[j+1]) +
                                    nw["YftR"][l]*x[i]*x[j]*sin(x[i+1]-x[j+1])
    end

    for l in nw["tobus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["fbus"])]-1)
        f += (-nw["YttI"][l])*x[i]^2 - nw["YtfI"][l]*x[i]*x[j]*cos(x[i+1]-x[j+1]) +
                                    nw["YtfR"][l]*x[i]*x[j]*sin(x[i+1]-x[j+1])
    end

    gen = 0.0
    for g in nw["busgen"][b]
        j = pf.var_pgqg_start + 2*(g-1) + 1
        gen += x[j]*baseMVA
    end

    f -= (gen-nw["bus"][b]["Qd"])/baseMVA
    return f
end

function eval_f(pf::PowerFlow, x::Vector{Float64}, F::Vector{Float64})
    for b=1:length(pf.nw["bus"])
        F[pf.equ_pg_start+(b-1)] = eval_f_real(pf, x, b)
    end
    for b=1:length(pf.nw["bus"])
        F[pf.equ_qg_start+(b-1)] = eval_f_reactive(pf, x, b)
    end
    return
end