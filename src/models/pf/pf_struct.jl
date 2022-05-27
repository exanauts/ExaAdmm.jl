mutable struct PowerFlow{T,TD}
    verbose::Int
    nw::Dict{String,Any}

    n_var::Int
    var_pgqg_start::Int
    var_vmva_start::Int
    equ_pg_start::Int
    equ_qg_start::Int
    jac_pg_start::Int
    jac_qg_start::Int
    nnz_jac::Int

    # Variable, function, and Jacobian for Newton-Raphson on CPU.
    x::Vector{Float64}
    xlo::Vector{Float64}
    xup::Vector{Float64}
    F::Vector{Float64}
    Jac::SparseMatrixCSC{Float64,Int}

    function PowerFlow{T,TD}(case::String; case_format="matpower", start_method="warm", use_scaling=false, verbose=1) where {T,TD<:AbstractArray{T}}
        nw = parse_matpower(case; case_format=case_format)
        return PowerFlow{T,TD}(nw; start_method=start_method, use_scaling=use_scaling, verbose=verbose)
    end

    function PowerFlow{T,TD}(nw::Dict{String,Any}; start_method="warm", verbose=1) where {T,TD<:AbstractArray{T}}
        pf = new()
        pf.verbose = verbose
        init_pf(pf, nw; start_method=start_method)
        return pf
    end
end
