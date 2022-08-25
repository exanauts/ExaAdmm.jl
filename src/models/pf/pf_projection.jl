function pf_projection(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol = mod.solution

    u_curr = zeros(mod.nvar)
    v_curr = zeros(mod.nvar)
    GenIdx = zeros(Int, length(mod.GenIdx))
    GenStart = zeros(Int, length(mod.GenStart))
    brBusIdx = zeros(Int, length(mod.brBusIdx))
    FrIdx = zeros(Int, length(mod.FrIdx))
    FrStart = zeros(Int, length(mod.FrStart))
    ToIdx = zeros(Int, length(mod.ToIdx))
    ToStart = zeros(Int, length(mod.ToStart))
    Pd = zeros(length(mod.grid_data.Pd))
    Qd = zeros(length(mod.grid_data.Qd))

    copyto!(u_curr, sol.u_curr)
    copyto!(v_curr, sol.v_curr)
    copyto!(GenIdx, mod.GenIdx)
    copyto!(GenStart, mod.GenStart)
    copyto!(brBusIdx, mod.brBusIdx)
    copyto!(FrIdx, mod.FrIdx)
    copyto!(FrStart, mod.FrStart)
    copyto!(ToIdx, mod.ToIdx)
    copyto!(ToStart, mod.ToStart)
    copyto!(Pd, mod.grid_data.Pd)
    copyto!(Qd, mod.grid_data.Qd)

    v_curr[mod.gen_start:mod.gen_start+2*mod.grid_data.ngen-1] .= u_curr[mod.gen_start:mod.gen_start+2*mod.grid_data.ngen-1]

    nw = env.data.nw
    pf = PowerFlow{Float64,Array{Float64}}(nw)

    # Extract voltage magnitudes and angles from u_curr.
    vm = Float64[0.0 for i=1:mod.nbus]
    va = Float64[0.0 for i=1:mod.nbus]
    for b=1:mod.nbus
        count = 0
        for k=FrStart[b]:FrStart[b+1]-1
            l = FrIdx[k]
            vm[b] += sqrt(u_curr[mod.line_start + 8*(l-1)+4])
            va[b] += u_curr[mod.line_start + 8*(l-1)+6]
            count += 1
        end
        for k=ToStart[b]:ToStart[b+1]-1
            l = ToIdx[k]
            vm[b] += sqrt(u_curr[mod.line_start + 8*(l-1)+5])
            va[b] += u_curr[mod.line_start + 8*(l-1)+7]
            count += 1
        end

        vm[b] /= count
        va[b] /= count
    end

    # Set an initial point for power flow solve.
    for b=1:mod.nbus
        pf.x[pf.var_vmva_start+2*(b-1)] = min(nw["bus"][b]["Vmax"], max(nw["bus"][b]["Vmin"], vm[b]))
        pf.x[pf.var_vmva_start+2*(b-1)+1] = va[b]
    end
    for g=1:mod.grid_data.ngen
        pf.x[pf.var_pgqg_start+2*(g-1)] = v_curr[mod.gen_start+2*(g-1)]
        pf.x[pf.var_pgqg_start+2*(g-1)+1] = v_curr[mod.gen_start+2*(g-1)+1]
    end

    solve_pf(pf)

    for b=1:mod.nbus
        vm[b] = pf.x[pf.var_vmva_start+2*(b-1)]
        va[b] = pf.x[pf.var_vmva_start+2*(b-1)+1]
        for k=FrStart[b]:FrStart[b+1]-1
            l = FrIdx[k]
            v_curr[mod.line_start + 8*(l-1)+4] = pf.x[pf.var_vmva_start+2*(b-1)]^2
            v_curr[mod.line_start + 8*(l-1)+6] = pf.x[pf.var_vmva_start+2*(b-1)+1]
        end
        for k=ToStart[b]:ToStart[b+1]-1
            l = ToIdx[k]
            v_curr[mod.line_start + 8*(l-1)+5] = pf.x[pf.var_vmva_start+2*(b-1)]^2
            v_curr[mod.line_start + 8*(l-1)+7] = pf.x[pf.var_vmva_start+2*(b-1)+1]
        end
    end

    # Recompute real/reactive power at slack bus and reactive power at PV buses.
    pv_idx = findall(x -> Int(x["type"]) == 2, pf.nw["bus"])
    slack_idx = findall(x -> Int(x["type"]) == 3, pf.nw["bus"])
    @assert length(slack_idx) == 1
    sbus = slack_idx[1]

    pg = Pd[sbus] / mod.baseMVA
    qg = Qd[sbus] / mod.baseMVA
    for k=FrStart[sbus]:FrStart[sbus+1]-1
        l = FrIdx[k]
        i = brBusIdx[1+2*(l-1)]
        j = brBusIdx[1+2*(l-1)+1]
        # pij
        pg += nw["YffR"][l]*vm[i]^2 + nw["YftR"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) + nw["YftI"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
        # qij
        qg += -nw["YffI"][l]*vm[i]^2 - nw["YftI"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) + nw["YftR"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
    end
    for k=ToStart[sbus]:ToStart[sbus+1]-1
        l = ToIdx[k]
        i = brBusIdx[1+2*(l-1)]
        j = brBusIdx[1+2*(l-1)+1]
        # pji
        pg += nw["YttR"][l]*vm[j]^2 + nw["YtfR"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) - nw["YtfI"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
        # qji
        qg += -nw["YttI"][l]*vm[j]^2 - nw["YtfI"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) - nw["YtfR"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
    end
    pg += nw["YshR"][sbus] * vm[sbus]^2
    qg -= nw["YshI"][sbus] * vm[sbus]^2

    ngen_bus = GenStart[sbus+1] - GenStart[sbus]
    for k=GenStart[sbus]:GenStart[sbus+1]-1
        g = GenIdx[k]
        v_curr[mod.gen_start+2*(g-1)] = pg / ngen_bus
        v_curr[mod.gen_start+2*(g-1)+1] = qg / ngen_bus
    end

    for b in pv_idx
        qg = Qd[b] / mod.baseMVA
        for k=FrStart[b]:FrStart[b+1]-1
            l = FrIdx[k]
            i = brBusIdx[1+2*(l-1)]
            j = brBusIdx[1+2*(l-1)+1]
            # qij
            qg += -nw["YffI"][l]*vm[i]^2 - nw["YftI"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) + nw["YftR"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
        end
        for k=ToStart[b]:ToStart[b+1]-1
            l = ToIdx[k]
            i = brBusIdx[1+2*(l-1)]
            j = brBusIdx[1+2*(l-1)+1]
            # qji
            qg += -nw["YttI"][l]*vm[j]^2 - nw["YtfI"][l]*vm[i]*vm[j]*cos(va[i] - va[j]) - nw["YtfR"][l]*vm[i]*vm[j]*sin(va[i] - va[j])
        end
        qg -= nw["YshI"][b] * vm[b]^2
        ngen_bus = GenStart[b+1] - GenStart[b]
        for k=GenStart[b]:GenStart[b+1]-1
            g = GenIdx[k]
            v_curr[mod.gen_start+2*(g-1)+1] = qg / ngen_bus
        end
    end

    copyto!(sol.v_curr, v_curr)
#    sol.u_curr .= sol.v_curr
#    sol.z_curr .= 0
    return
end