function solve_acopf_mpec_ipopt(case::String; case_format="matpower",
                                enable_fq=false, enable_vm=false, eps_fq=0.0, eps_vm=0.0)
    VI = Array{Int}; VD = Array{Float64}
    data = opf_loaddata(case; VI=VI, VD=VD, case_format=case_format)
    ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=VI, VD=VD)...)

    mod = JuMP.Model(Ipopt.Optimizer)
    ngen = length(data.generators)
    nline = length(data.lines)
    nbus = length(data.buses)

    rateA = Float64[data.lines[l].rateA == 0.0 ? 1e5 : data.lines[l].rateA for l=1:nline]
    pv_idx = Int64[i for i=1:nbus if data.buses[i].bustype in [2,3]]
    nbus_pv = length(pv_idx)
    pv_gen_idx = Int64[data.BusGenerators[pv_idx[i]][1] for i=1:nbus_pv]

    R = 0.04
    pg_setpoint = Float64[(data.generators[g].Pmax+data.generators[g].Pmin)/2 for g=1:ngen]
    alpha = Float64[-(1/R)*data.generators[g].Pmax for g=1:ngen]

    sum_Pd = sum(data.buses[b].Pd/data.baseMVA for b=1:nbus)
    sum_pg_setpoint = sum(pg_setpoint[g] for g=1:ngen)
    @printf("sum_Pd = %.6e  sum_pg_setpoint = %.6e\n", sum_Pd, sum_pg_setpoint)

    @variable(mod, data.generators[g].Pmin <= Pg[g=1:ngen] <= data.generators[g].Pmax, start=data.generators[g].Pg)
    @variable(mod, data.generators[g].Qmin <= Qg[g=1:ngen] <= data.generators[g].Qmax, start=data.generators[g].Qg)
    @variable(mod, data.buses[b].Vmin <= Vm[b=1:nbus] <= data.buses[b].Vmax, start=data.buses[b].Vm)
    @variable(mod, -2*pi <= Va[b=1:nbus] <= 2*pi, start=0.0)

    @variable(mod, ReactiveSlack[b=1:nbus], start=0.0)
    @variable(mod, DeltaFreq, start=0.0)

    if enable_fq

        @variable(mod, Fg_w[i=1:ngen] >= 0, start=0.0)
        @variable(mod, Fg_v[i=1:ngen] >= 0, start=0.0)
        @constraint(mod, Fg_primary[i=1:ngen],
                    Pg[i] - (pg_setpoint[i] + alpha[i]*DeltaFreq) == Fg_w[i] - Fg_v[i])
        @NLconstraint(mod, Fg_complementarity_w[i=1:ngen],
                    (Pg[i] - data.generators[i].Pmin)*(Fg_w[i]) <= eps_fq
        )
        @NLconstraint(mod, Fg_complementarity_v[i=1:ngen],
                    (data.generators[i].Pmax - Pg[i])*(Fg_v[i]) <= eps_fq
        )
    end

    if enable_vm
        @variable(mod, Vg_w[i=1:nbus_pv] >= 0, start=0.0)
        @variable(mod, Vg_v[i=1:nbus_pv] >= 0, start=0.0)
        @constraint(mod, Vg_setpoint[i=1:nbus_pv],
            Vm[pv_idx[i]] - 1.0 == Vg_w[i] - Vg_v[i]
        )
        @NLconstraint(mod, Vg_complementarity_w[i=1:nbus_pv],
            (Qg[pv_gen_idx[i]] - data.generators[pv_gen_idx[i]].Qmin)*(Vg_w[i]) <= eps_vm
        )
        @NLconstraint(mod, Vg_complementarity_v[i=1:nbus_pv],
            (data.generators[pv_gen_idx[i]].Qmax - Qg[pv_gen_idx[i]])*(Vg_v[i]) <= eps_vm
        )
    end

    set_lower_bound(Va[data.bus_ref], 0.0)
    set_upper_bound(Va[data.bus_ref], 0.0)

    @NLexpression(mod, pij[l=1:nline],
        ybus.YffR[l]*(Vm[data.BusIdx[data.lines[l].from]]^2)
        + ybus.YftR[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                      cos(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
        + ybus.YftI[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                       sin(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
    )

    @NLexpression(mod, pji[l=1:nline],
        ybus.YttR[l]*(Vm[data.BusIdx[data.lines[l].to]]^2)
        + ybus.YtfR[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                      cos(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
        - ybus.YtfI[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                       sin(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
    )

    @NLexpression(mod, qij[l=1:nline],
        - ybus.YffI[l]*(Vm[data.BusIdx[data.lines[l].from]]^2)
        - ybus.YftI[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                      cos(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
        + ybus.YftR[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                       sin(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
    )

    @NLexpression(mod, qji[l=1:nline],
        - ybus.YttI[l]*(Vm[data.BusIdx[data.lines[l].to]]^2)
        - ybus.YtfI[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                      cos(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
        - ybus.YtfR[l]*((Vm[data.BusIdx[data.lines[l].from]]*Vm[data.BusIdx[data.lines[l].to]])*
                       sin(Va[data.BusIdx[data.lines[l].from]] - Va[data.BusIdx[data.lines[l].to]])
                     )
    )

    @NLconstraint(mod, real_power_balance[b=1:nbus],
        (sum(data.baseMVA*Pg[g] for g in data.BusGenerators[b])-data.buses[b].Pd)/data.baseMVA
        == ybus.YshR[b]*Vm[b]^2 +
            sum(pij[l] for l in data.FromLines[b]) + sum(pji[l] for l in data.ToLines[b])
    )

    @NLconstraint(mod, reactive_power_balance[b=1:nbus],
        (sum(data.baseMVA*Qg[g] for g in data.BusGenerators[b])-data.buses[b].Qd)/data.baseMVA #+ ReactiveSlack[b]
        == -ybus.YshI[b]*Vm[b]^2 +
            sum(qij[l] for l in data.FromLines[b]) + sum(qji[l] for l in data.ToLines[b])
    )
#=
    @NLconstraint(mod, ij_linelimit[l=1:nline],
        pij[l]^2 + qij[l]^2 <= (rateA[l]/data.baseMVA)^2
    )

    @NLconstraint(mod, ji_linelimit[l=1:nline],
        pji[l]^2 + qji[l]^2 <= (rateA[l]/data.baseMVA)^2
    )
=#
#=
    @NLobjective(mod, Min,
        sum(data.generators[g].coeff[data.generators[g].n-2]*(data.baseMVA*Pg[g])^2
            + data.generators[g].coeff[data.generators[g].n-1]*(data.baseMVA*Pg[g])
            + data.generators[g].coeff[data.generators[g].n] for g=1:ngen)
#        + 1e5*sum(ReactiveSlack[b]^2 for b=1:nbus)
    )
=#
    optimize!(mod)

    status = termination_status(mod)
    if status == OPTIMAL || status == LOCALLY_SOLVED
        for b=1:nbus
            @printf("[%3d] bus type = %d reactive_slack = %12.6e\n", b, data.buses[b].bustype, value(ReactiveSlack[b]))
        end
#=
        for l=1:nline
            ij_lineflow = value(pij[l])^2 + value(qij[l])^2
            ji_lineflow = value(pji[l])^2 + value(qji[l])^2
            if (ij_lineflow > (rateA[l]/data.baseMVA)^2)
                @printf("[%3d] ij_lineflow violated: %12.6e > %12.6e\n", ij_lineflow, (rateA[l]/data.baseMVA)^2)
            end
            if (ji_lineflow > (rateA[l]/data.baseMVA)^2)
                @printf("[%3d] ji_lineflow violated: %12.6e > %12.6e\n", ji_lineflow, (rateA[l]/data.baseMVA)^2)
            end
        end
=#
        if enable_vm
            @printf("Complementarity between VM and QG:\n")
            for i=1:nbus_pv
                @printf("Vm[%3d] = %12.6e  % 12.6e  % 12.6e\n",
                    pv_idx[i], value(Vm[pv_idx[i]]),
                    (value(Qg[pv_gen_idx[i]]) - data.generators[pv_gen_idx[i]].Qmin)*(value(Vg_w[i])),
                    (data.generators[pv_gen_idx[i]].Qmax - value(Qg[pv_gen_idx[i]]))*(value(Vg_v[i]))
                )
            end
        end

        if enable_fq
            @printf("Complementarity between PG and DELTA FREQ:\n")
            @printf("DELTA FREQ = %12.6e\n", value(DeltaFreq))
            for g=1:ngen
                delta_Pg = pg_setpoint[g] + alpha[g]*value(DeltaFreq)
                @printf("[%3d] %12.6e <= PG = %12.6e <= %12.6e",
                        g, data.generators[g].Pmin, value(Pg[g]), data.generators[g].Pmax)
                if (delta_Pg < value(Pg[g]) - 1e-6)
                    @printf(" > ")
                elseif (delta_Pg > value(Pg[g]) + 1e-6)
                    @printf(" < ")
                else
                    @printf(" = ")
                end
                @printf("%12.6e\n", delta_Pg)
            end
        end
    end

    return data, mod
end