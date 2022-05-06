function init_pf(pf::PowerFlow, nw::Dict{String,Any}; start_method="warm")
    pf.nw = nw

    # The number of Jacobian entries of real power balance equations:
    #
    # For each bus,
    #  - dg/dvm_i, dg/dva_i so 2 nonzeros are added.
    #  - dg/dvm{to,fr} and dg/dva{to,fr} so 2*nbranch are added.
    #    By symmetric, we have additional 2*nbranch.
    #  - generator also has nonzero values.
    #
    # The resulting number is multiplied by 2 to include reactive power.

    pf.nnz_jac = 2*length(nw["bus"]) + 4*length(nw["branch"]) + length(nw["gen"])
    pf.nnz_jac *= 2

    pf.n_var = 2*length(nw["bus"]) + 2*length(nw["gen"])
    pf.var_vmva_start = 1
    pf.var_pgqg_start = pf.var_vmva_start + 2*length(nw["bus"])
    pf.equ_pg_start = 1
    pf.equ_qg_start = pf.equ_pg_start + length(nw["bus"])
    pf.jac_pg_start = 1
    pf.jac_qg_start = pf.jac_pg_start + div(pf.nnz_jac, 2)

    pf.x = zeros(pf.n_var)
    pf.xlo = -Inf*ones(pf.n_var)
    pf.xup = Inf*ones(pf.n_var)
    pf.F = zeros(2*length(nw["bus"]))
    I = zeros(Int, pf.nnz_jac)
    J = zeros(Int, pf.nnz_jac)
    get_jac_struct(pf, I, J)
    pf.Jac = sparse(I, J, zeros(pf.nnz_jac))

    for i=1:length(nw["bus"])
        pf.xlo[pf.var_vmva_start+2*(i-1)] = nw["bus"][i]["Vmin"]
        pf.xup[pf.var_vmva_start+2*(i-1)] = nw["bus"][i]["Vmax"]
    end
    for g=1:length(nw["gen"])
        pf.xlo[pf.var_pgqg_start+2*(g-1)] = nw["gen"][g]["Pmin"]
        pf.xlo[pf.var_pgqg_start+2*(g-1)+1] = nw["gen"][g]["Qmin"]
        pf.xup[pf.var_pgqg_start+2*(g-1)] = nw["gen"][g]["Pmax"]
        pf.xup[pf.var_pgqg_start+2*(g-1)+1] = nw["gen"][g]["Qmax"]
    end

    init_start_x(pf, pf.x; start_method=start_method)

    return
end

function init_start_x_warm(pf::PowerFlow, x::Vector{Float64})
    nw = pf.nw
    for i=1:length(nw["bus"])
        x[pf.var_vmva_start+2*(i-1)] = max(nw["bus"][i]["Vmin"], min(nw["bus"][i]["Vmax"], nw["bus"][i]["Vm"]))
        x[pf.var_vmva_start+2*(i-1)+1] = nw["bus"][i]["Va"]
    end
    for g=1:length(nw["gen"])
        x[pf.var_pgqg_start+2*(g-1)] = max(nw["gen"][g]["Pmin"], min(nw["gen"][g]["Pmax"], nw["gen"][g]["Pg"]))
        x[pf.var_pgqg_start+2*(g-1)+1] = max(nw["gen"][g]["Qmin"], min(nw["gen"][g]["Qmax"], nw["gen"][g]["Qg"]))
    end
end

function init_start_x_flat(pf::PowerFlow, x::Vector{Float64})
    nw = pf.nw
    for i=1:length(nw["bus"])
        x[pf.var_vmva_start+2*(i-1)] = (nw["bus"][i]["Vmin"]+nw["bus"][i]["Vmax"])/2
        x[pf.var_vmva_start+2*(i-1)+1] = 0.0
    end
    for g=1:length(nw["gen"])
        x[pf.var_pgqg_start+2*(g-1)] = (nw["gen"][g]["Pmin"]+nw["gen"][g]["Pmax"])/2
        x[pf.var_pgqg_start+2*(g-1)+1] = (nw["gen"][g]["Qmin"]+nw["gen"][g]["Qmax"])/2
    end
end

function init_start_x(pf::PowerFlow, x::Vector{Float64}; start_method="warm")
    if start_method == "warm"
        init_start_x_warm(pf, x)
    elseif start_method == "flat"
        init_start_x_flat(pf, x)
    else
        error("Unknown start_method ", start_method)
    end
end