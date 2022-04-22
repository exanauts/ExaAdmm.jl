"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct ModelAcopf{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    solution::AbstractSolution{T,TD}

    # Used for multiple dispatch for multi-period case.
    gen_solution::AbstractSolution{T,TD}

    n::Int
    ngen::Int
    nline::Int
    nbus::Int
    nvar::Int

    gen_start::Int
    line_start::Int

    baseMVA::T
    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    pgmin_curr::TD   # taking ramping into account for rolling horizon
    pgmax_curr::TD   # taking ramping into account for rolling horizon
    ramp_rate::TD
    c2::TD
    c1::TD
    c0::TD
    YshR::TD
    YshI::TD
    YffR::TD
    YffI::TD
    YftR::TD
    YftI::TD
    YttR::TD
    YttI::TD
    YtfR::TD
    YtfI::TD
    FrVmBound::TD
    ToVmBound::TD
    FrVaBound::TD
    ToVaBound::TD
    rateA::TD
    FrStart::TI
    FrIdx::TI
    ToStart::TI
    ToIdx::TI
    GenStart::TI
    GenIdx::TI
    Pd::TD
    Qd::TD
    Vmin::TD
    Vmax::TD

    chg_min::TD
    chg_max::TD
    energy_min::TD
    energy_max::TD
    eta_chg::TD
    eta_dischg::TD

    StorageStart::TI
    StorageIdx::TI

    membuf::TM      # memory buffer for line kernel
    gen_membuf::TM  # memory buffer for generator kernel

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    function ModelAcopf{T,TD,TI,TM}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        return new{T,TD,TI,TM}()
    end

    function ModelAcopf{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}; ramp_ratio=0.02) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()

        model.baseMVA = env.data.baseMVA
        model.n = (env.use_linelimit == true) ? 6 : 4
        model.ngen = length(env.data.generators)
        model.nline = length(env.data.lines)
        model.nbus = length(env.data.buses)
        model.nline_padded = model.nline

        # Memory space is padded for the lines as a multiple of # processes.
        if env.use_mpi
            nprocs = MPI.Comm_size(env.comm)
            model.nline_padded = nprocs * div(model.nline, nprocs, RoundUp)
        end

        model.nvar = 2*model.ngen + 8*model.nline
        model.nvar_padded = model.nvar + 8*(model.nline_padded - model.nline)
        model.gen_start = 1
        model.line_start = 2*model.ngen + 1
        model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1, model.c0 = get_generator_data(env.data; use_gpu=env.use_gpu)
        model.YshR, model.YshI, model.YffR, model.YffI, model.YftR, model.YftI,
            model.YttR, model.YttI, model.YtfR, model.YtfI,
            model.FrVmBound, model.ToVmBound,
            model.FrVaBound, model.ToVaBound, model.rateA = get_branch_data(env.data; use_gpu=env.use_gpu, tight_factor=env.tight_factor)
        model.FrStart, model.FrIdx, model.ToStart, model.ToIdx, model.GenStart, model.GenIdx, model.Pd, model.Qd, model.Vmin, model.Vmax = get_bus_data(env.data; use_gpu=env.use_gpu)
        model.brBusIdx = get_branch_bus_index(env.data; use_gpu=env.use_gpu)
        model.chg_min, model.chg_max, model.energy_min, model.energy_max, model.eta_chg, model.eta_dischg = get_storage_data(env.data; use_gpu=env.use_gpu)
        model.StorageIdx, model.StorageStart = get_bus_storage_index(env.data; use_gpu=env.use_gpu)

        model.pgmin_curr = TD(undef, model.ngen)
        model.pgmax_curr = TD(undef, model.ngen)
        copyto!(model.pgmin_curr, model.pgmin)
        copyto!(model.pgmax_curr, model.pgmax)

        model.ramp_rate = TD(undef, model.ngen)
        model.ramp_rate .= ramp_ratio.*model.pgmax

        if env.params.obj_scale != 1.0
            model.c2 .*= env.params.obj_scale
            model.c1 .*= env.params.obj_scale
            model.c0 .*= env.params.obj_scale
        end

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.ngen + 8*model.nline
        model.nvar_u_padded = model.nvar_u + 8*(model.nline_padded - model.nline)
        model.nvar_v = 2*model.ngen + 4*model.nline + 2*model.nbus
        model.bus_start = 2*model.ngen + 4*model.nline + 1
        if env.use_twolevel
            model.nvar = model.nvar_u + model.nvar_v
            model.nvar_padded = model.nvar_u_padded + model.nvar_v
        end

        # Memory space is allocated based on the padded size.
        model.solution = ifelse(env.use_twolevel,
            SolutionTwoLevel{T,TD}(model.nvar_padded, model.nvar_v, model.nline_padded),
            SolutionOneLevel{T,TD}(model.nvar_padded))
        init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va)
        model.gen_solution = EmptyGeneratorSolution{T,TD}()

        model.membuf = TM(undef, (31, model.nline))
        fill!(model.membuf, 0.0)
        model.membuf[29,:] .= model.rateA

        model.info = IterationInformation{ComponentInformation}()

        return model
    end
end

"""
This is to share power network data between models. Some fields that could be modified are deeply copied.
"""
function Base.copy(ref::ModelAcopf{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
    model = ModelAcopf{T,TD,TI,TM}()

    model.solution = copy(ref.solution)
    model.gen_solution = copy(ref.gen_solution)
    model.info = copy(ref.info)

    model.n = ref.n
    model.ngen = ref.ngen
    model.nline = ref.nline
    model.nbus = ref.nbus
    model.nvar = ref.nvar

    model.gen_start = ref.gen_start
    model.line_start = ref.line_start

    model.baseMVA = ref.baseMVA
    model.pgmin = ref.pgmin
    model.pgmax = ref.pgmax
    model.qgmin = ref.qgmin
    model.qgmax = ref.qgmax
    model.pgmin_curr = copy(ref.pgmin_curr)
    model.pgmax_curr = copy(ref.pgmax_curr)
    model.ramp_rate = ref.ramp_rate
    model.c2 = ref.c2
    model.c1 = ref.c1
    model.c0 = ref.c0
    model.YshR = ref.YshR
    model.YshI = ref.YshI
    model.YffR = ref.YffR
    model.YffI = ref.YffI
    model.YftR = ref.YftR
    model.YftI = ref.YftI
    model.YttR = ref.YttR
    model.YttI = ref.YttI
    model.YtfR = ref.YtfR
    model.YtfI = ref.YtfI
    model.FrVmBound = ref.FrVmBound
    model.ToVmBound = ref.ToVmBound
    model.FrVaBound = ref.FrVaBound
    model.ToVaBound = ref.ToVaBound
    model.rateA = ref.rateA
    model.FrStart = ref.FrStart
    model.FrIdx = ref.FrIdx
    model.ToStart = ref.ToStart
    model.ToIdx = ref.ToIdx
    model.GenStart = ref.GenStart
    model.GenIdx = ref.GenIdx
    model.Pd = copy(ref.Pd)
    model.Qd = copy(ref.Qd)
    model.Vmin = ref.Vmin
    model.Vmax = ref.Vmax

    model.membuf = copy(ref.membuf)

    model.nvar_u = ref.nvar_u
    model.nvar_v = ref.nvar_v
    model.bus_start = ref.bus_start
    model.brBusIdx = ref.brBusIdx

    model.nline_padded = ref.nline_padded
    model.nvar_padded = ref.nvar_padded

    return model
end
