function get_jac_struct(pf::PowerFlow,
    jac_start::Int, row_start::Int, col_start::Int, pgqg::Int,
    I::Vector{Int}, J::Vector{Int})

    nnz = jac_start
    for i=1:length(pf.nw["bus"])
        row = row_start + (i-1)
        I[nnz] = row # P_i or Q_i
        J[nnz] = col_start + 2*(i-1) # vm_i
        nnz += 1
        I[nnz] = row
        J[nnz] = col_start + 2*(i-1) + 1  # va_i
        nnz += 1

        for l in pf.nw["frombus"][i]
            j = pf.nw["bus2idx"][Int(pf.nw["branch"][l]["tbus"])]
            I[nnz] = row
            J[nnz] = col_start + 2*(j-1) # vm_j
            nnz += 1
            I[nnz] = row
            J[nnz] = col_start + 2*(j-1) + 1 # va_j
            nnz += 1
        end

        for l in pf.nw["tobus"][i]
            j = pf.nw["bus2idx"][Int(pf.nw["branch"][l]["fbus"])]
            I[nnz] = row
            J[nnz] = col_start + 2*(j-1) # vm_j
            nnz += 1
            I[nnz] = row
            J[nnz] = col_start + 2*(j-1) + 1 # va_j
            nnz += 1
        end

        for g in pf.nw["busgen"][i]
            I[nnz] = row
            J[nnz] = pf.var_pgqg_start + 2*(g-1) + pgqg  # p_gi or q_gi
            nnz += 1
        end
    end
end

function get_jac_struct(pf::PowerFlow, I::Vector{Int}, J::Vector{Int})
    @assert pf.nnz_jac == length(I) && pf.nnz_jac == length(J)

    # Real power
    get_jac_struct(pf, pf.jac_pg_start, pf.equ_pg_start, pf.var_vmva_start, 0, I, J)
    # Reactive power
    get_jac_struct(pf, pf.jac_qg_start, pf.equ_qg_start, pf.var_vmva_start, 1, I, J)

    return
end

function eval_jac_real(pf::PowerFlow, x::Vector{Float64}, Jac::SparseMatrixCSC{Float64,Int}, b::Int)
    nw = pf.nw

    r = pf.equ_pg_start + (b-1)
    i = pf.var_vmva_start + 2*(b-1)

    Jac[r,i] += (2*nw["YshR"][b])*x[i]
    for l in nw["frombus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["tbus"])]-1)
        cos_ij = cos(x[i+1]-x[j+1])
        sin_ij = sin(x[i+1]-x[j+1])
        val1 = nw["YftR"][l]*cos_ij + nw["YftI"][l]*sin_ij
        val2 = (-nw["YftR"][l])*sin_ij + nw["YftI"][l]*cos_ij
        val2 *= x[i]*x[j]

        Jac[r,i] += (2*nw["YffR"][l])*x[i] + val1*x[j] # dP/dvm_i
        Jac[r,i+1] += val2 # dP/dva_i
        Jac[r,j] += val1*x[i] # dP/dvm_j
        Jac[r,j+1] += -val2 # dP/dva_j
    end

    for l in nw["tobus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["fbus"])]-1)
        cos_ij = cos(x[i+1]-x[j+1])
        sin_ij = sin(x[i+1]-x[j+1])
        val1 = nw["YtfR"][l]*cos_ij + nw["YtfI"][l]*sin_ij
        val2 = (-nw["YtfR"][l])*sin_ij + nw["YtfI"][l]*cos_ij
        val2 *= x[i]*x[j]

        Jac[r,i] += (2*nw["YttR"][l])*x[i] + val1*x[j] # dP/dvm_i
        Jac[r,i+1] += val2 # dP/dva_i
        Jac[r,j] += val1*x[i] # dP/dvm_j
        Jac[r,j+1] += -val2 # dP/dva_j
    end

    for g in nw["busgen"][b]
        j = pf.var_pgqg_start + 2*(g-1)
        Jac[r,j] = -1.0
    end
    return
end

function eval_jac_reactive(pf::PowerFlow, x::Vector{Float64}, Jac::SparseMatrixCSC{Float64,Int}, b::Int)
    nw = pf.nw

    r = pf.equ_qg_start + (b-1)
    i = pf.var_vmva_start + 2*(b-1)

    Jac[r,i] += (-2*nw["YshI"][b])*x[i]
    for l in pf.nw["frombus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["tbus"])]-1)
        cos_ij = cos(x[i+1]-x[j+1])
        sin_ij = sin(x[i+1]-x[j+1])
        val1 = (-nw["YftI"][l])*cos_ij + nw["YftR"][l]*sin_ij
        val2 = nw["YftI"][l]*sin_ij + nw["YftR"][l]*cos_ij
        val2 *= x[i]*x[j]

        Jac[r,i] += (-2*nw["YffI"][l])*x[i] + val1*x[j] # dQ/dvm_i
        Jac[r,i+1] += val2 # dQ/dva_i
        Jac[r,j] += val1*x[i] # dQ/dvm_j
        Jac[r,j+1] += -val2 # dQ/dva_j
    end

    for l in pf.nw["tobus"][b]
        j = pf.var_vmva_start + 2*(nw["bus2idx"][Int(nw["branch"][l]["fbus"])]-1)
        cos_ij = cos(x[i+1]-x[j+1])
        sin_ij = sin(x[i+1]-x[j+1])
        val1 = (-nw["YtfI"][l])*cos_ij + nw["YtfR"][l]*sin_ij
        val2 = nw["YtfI"][l]*sin_ij + nw["YtfR"][l]*cos_ij
        val2 *= x[i]*x[j]

        Jac[r,i] += (-2*nw["YttI"][l])*x[i] + val1*x[j] # dQ/dvm_i
        Jac[r,i+1] += val2 # dQ/dva_i
        Jac[r,j] += val1*x[i] # dQ/dvm_j
        Jac[r,j+1] += -val2 # dQ/dva_j
    end

    for g in pf.nw["busgen"][b]
        j = pf.var_pgqg_start + 2*(g-1) + 1
        Jac[r,j] = -1.0
    end
    return
end

function eval_jac(pf::PowerFlow, x::Vector{Float64}, Jac::SparseMatrixCSC{Float64,Int})
    fill!(Jac, 0.0)
    for b=1:length(pf.nw["bus"])
        eval_jac_real(pf, x, Jac, b)
    end
    for b=1:length(pf.nw["bus"])
        eval_jac_reactive(pf, x, Jac, b)
    end
    return
end

