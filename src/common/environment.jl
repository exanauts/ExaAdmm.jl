"""
    Parameters

This contains the parameters used in ADMM algorithm.
"""
mutable struct Parameters
    mu_max::Float64 # Augmented Lagrangian
    max_auglag::Int # Augmented Lagrangian
    ABSTOL::Float64
    RELTOL::Float64

    rho_max::Float64    # TODO: not used
    rho_min_pq::Float64 # TODO: not used
    rho_min_w::Float64  # TODO: not used
    eps_rp::Float64     # TODO: not used
    eps_rp_min::Float64 # TODO: not used
    rt_inc::Float64     # TODO: not used
    rt_dec::Float64     # TODO: not used
    eta::Float64        # TODO: not used
    verbose::Int

    # MPI implementation
    shift_lines::Int

    # Two-Level ADMM
    initial_beta::Float64
    beta::Float64
    inc_c::Float64
    theta::Float64
    outer_eps::Float64
    shmem_size::Int
    gen_shmem_size::Int
    Kf::Int             # TODO: not used
    Kf_mean::Int        # TODO: not used
    MAX_MULTIPLIER::Float64
    DUAL_TOL::Float64

    outer_iterlim::Int
    inner_iterlim::Int
    scale::Float64
    obj_scale::Float64

    function Parameters()
        par = new()
        par.mu_max = 1e8
        par.rho_max = 1e6
        par.rho_min_pq = 5.0
        par.rho_min_w = 5.0
        par.eps_rp = 1e-4
        par.eps_rp_min = 1e-5
        par.rt_inc = 2.0
        par.rt_dec = 2.0
        par.eta = 0.99
        par.max_auglag = 50
        par.ABSTOL = 1e-6
        par.RELTOL = 1e-5
        par.verbose = 1
        par.shift_lines = 0
        par.initial_beta = 1e3
        par.beta = 1e3
        par.inc_c = 6.0
        par.theta = 0.8
        par.outer_eps = 2*1e-4
        par.shmem_size = 0
        par.gen_shmem_size = 0
        par.Kf = 100
        par.Kf_mean = 10
        par.MAX_MULTIPLIER = 1e12
        par.DUAL_TOL = 1e-8

        par.outer_iterlim = 20
        par.inner_iterlim = 1000
        par.scale = 1e-4
        par.obj_scale = 1.0

        return par
    end
end

abstract type AbstractAdmmEnv{T,TD,TI,TM} end

"""
    AdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct AdmmEnv{T,TD,TI,TM} <: AbstractAdmmEnv{T,TD,TI,TM}
    case::String
    data::OPFData
    load::Load{TM}
    initial_rho_pq::Float64
    initial_rho_va::Float64
    tight_factor::Float64
    horizon_length::Int
    use_gpu::Bool
    use_linelimit::Bool
    use_twolevel::Bool
    use_mpi::Bool
    use_projection::Bool
    load_specified::Bool
    gpu_no::Int
    comm::MPI.Comm

    params::Parameters
#    model::AbstractOPFModel{T,TD,TI}
#    membuf::TM # was param

    function AdmmEnv{T,TD,TI,TM}(
        case::String, rho_pq::Float64, rho_va::Float64;
        case_format="matpower",
        use_gpu=false, use_linelimit=false, use_twolevel=false, use_mpi=false, use_projection=false,
        gpu_no::Int=1, verbose::Int=1, tight_factor=1.0,
        horizon_length=1, load_prefix::String="", comm::MPI.Comm=MPI.COMM_WORLD
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        env = new{T,TD,TI,TM}()
        env.case = case
        env.data = opf_loaddata(env.case; VI=TI, VD=TD, case_format=case_format)
        env.initial_rho_pq = rho_pq
        env.initial_rho_va = rho_va
        env.tight_factor = tight_factor
        env.use_gpu = use_gpu
        env.use_linelimit = use_linelimit
        env.use_mpi = use_mpi
        env.use_projection = use_projection
        env.gpu_no = gpu_no
        env.use_twolevel = use_twolevel
        env.load_specified = false
        env.comm = comm

        env.params = Parameters()
        env.params.verbose = verbose

        env.horizon_length = horizon_length
        if !isempty(load_prefix)
            env.load = get_load(load_prefix; use_gpu=use_gpu)
            @assert size(env.load.pd) == size(env.load.qd)
            @assert size(env.load.pd,2) >= horizon_length && size(env.load.qd,2) >= horizon_length
            env.load_specified = true
        end

        return env
    end
end

abstract type AbstractSolution{T,TD} end

struct EmptyGeneratorSolution{T,TD} <: AbstractSolution{T,TD}
    function EmptyGeneratorSolution{T,TD}() where {T, TD<:AbstractArray{T}}
        return new{T,TD}()
    end
end

function Base.copy(ref::EmptyGeneratorSolution{T,TD}) where {T,TD<:AbstractArray{T}}
    return EmptyGeneratorSolution{T,TD}()
end

"""
    SolutionOneLevel{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct SolutionOneLevel{T,TD} <: AbstractSolution{T,TD}
    u_curr::TD
    v_curr::TD
    l_curr::TD
    u_prev::TD
    v_prev::TD
    l_prev::TD
    rho::TD
    rd::TD
    rp::TD
    rp_prev::TD
    z_outer::TD    # used only for the two-level formulation
    z_curr::TD     # used only for the two-level formulation
    z_prev::TD     # used only for the two-level formulation
    lz::TD         # used only for the two-level formulation
    Ax_plus_By::TD # used only for the two-level formulation
    overall_time::T
    max_viol_except_line::T
    max_line_viol_rateA::T
    cumul_iters::Int
    status::Symbol

    function SolutionOneLevel{T,TD}(nvar::Int) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, nvar), # u_curr
            TD(undef, nvar), # v_curr
            TD(undef, nvar), # l_curr
            TD(undef, nvar), # u_prev
            TD(undef, nvar), # v_prev
            TD(undef, nvar), # l_prev
            TD(undef, nvar), # rho
            TD(undef, nvar), # rd
            TD(undef, nvar), # rp
            TD(undef, nvar), # rp_prev
            TD(undef, nvar), # z_outer
            TD(undef, nvar), # z_curr
            TD(undef, nvar), # z_prev
            TD(undef, nvar), # lz
            TD(undef, nvar), # Ax_plus_By
            Inf,
            Inf,
            Inf,
            0,
            :NotSpecified
        )

        fill!(sol, 0.0)
        return sol
    end
end


function Base.fill!(sol::SolutionOneLevel, val)
    fill!(sol.u_curr, val)
    fill!(sol.v_curr, val)
    fill!(sol.l_curr, val)
    fill!(sol.u_prev, val)
    fill!(sol.v_prev, val)
    fill!(sol.l_prev, val)
    fill!(sol.rho, val)
    fill!(sol.rd, val)
    fill!(sol.rp, val)
    fill!(sol.rp_prev, val)
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.lz, val)
    fill!(sol.Ax_plus_By, val)
end

function Base.copy(ref::SolutionOneLevel{T,TD}) where {T,TD<:AbstractArray{T}}
    nvar = length(ref.u_curr)
    sol = SolutionOneLevel{T,TD}(nvar)

    copyto!(sol.u_curr, ref.u_curr)
    copyto!(sol.v_curr, ref.v_curr)
    copyto!(sol.l_curr, ref.l_curr)
    copyto!(sol.u_prev, ref.u_prev)
    copyto!(sol.v_prev, ref.v_prev)
    copyto!(sol.l_prev, ref.l_prev)
    copyto!(sol.rho, ref.rho)
    copyto!(sol.rd, ref.rd)
    copyto!(sol.rp, ref.rp)
    copyto!(sol.rp_prev, ref.rp_prev)
    copyto!(sol.z_outer, ref.z_outer)
    copyto!(sol.z_curr, ref.z_curr)
    copyto!(sol.z_prev, ref.z_prev)
    copyto!(sol.lz, ref.lz)
    copyto!(sol.Ax_plus_By, ref.Ax_plus_By)
    sol.overall_time = ref.overall_time
    sol.max_viol_except_line = ref.max_viol_except_line
    sol.max_line_viol_rateA = ref.max_line_viol_rateA
    sol.cumul_iters = ref.cumul_iters
    sol.status = ref.status
    return sol
end

"""
    SolutionTwoLevel{T,TD}

This contains the solutions of ACOPF model instance for two-level ADMM algorithm,
    including the ADMM parameter rho.
"""
mutable struct SolutionTwoLevel{T,TD} <: AbstractSolution{T,TD}
    x_curr::TD
    xbar_curr::TD
    z_outer::TD
    z_curr::TD
    z_prev::TD
    l_curr::TD
    lz::TD
    rho::TD
    rp::TD
    rd::TD
    rp_old::TD
    Ax_plus_By::TD
    wRIij::TD

    function SolutionTwoLevel{T,TD}() where {T, TD<:AbstractArray{T}}
        return new{T,TD}()
    end

    function SolutionTwoLevel{T,TD}(nvar::Int, nvar_v::Int, nline::Int) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, nvar),      # x_curr
            TD(undef, nvar_v),    # xbar_curr
            TD(undef, nvar),      # z_outer
            TD(undef, nvar),      # z_curr
            TD(undef, nvar),      # z_prev
            TD(undef, nvar),      # l_curr
            TD(undef, nvar),      # lz
            TD(undef, nvar),      # rho
            TD(undef, nvar),      # rp
            TD(undef, nvar),      # rd
            TD(undef, nvar),      # rp_old
            TD(undef, nvar),      # Ax_plus_By
            TD(undef, 2*nline)    # wRIij
        )

        fill!(sol, 0.0)

        return sol
    end
end

function Base.fill!(sol::SolutionTwoLevel, val)
    fill!(sol.x_curr, val)
    fill!(sol.xbar_curr, val)
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.l_curr, val)
    fill!(sol.lz, val)
    fill!(sol.rho, val)
    fill!(sol.rp, val)
    fill!(sol.rd, val)
    fill!(sol.rp_old, val)
    fill!(sol.Ax_plus_By, val)
    fill!(sol.wRIij, val)
end

function Base.copy(ref::SolutionTwoLevel{T,TD}) where {T, TD<:AbstractArray{T}}
    sol = SolutionTwoLevel{T,TD}()
    sol.x_curr = TD(undef, length(ref.x_curr))
    sol.xbar_curr = TD(undef, length(ref.xbar_curr))
    sol.z_outer = TD(undef, length(ref.z_outer))
    sol.z_curr = TD(undef, length(ref.z_curr))
    sol.z_prev = TD(undef, length(ref.z_prev))
    sol.l_curr = TD(undef, length(ref.l_curr))
    sol.lz = TD(undef, length(ref.lz))
    sol.rho = TD(undef, length(ref.rho))
    sol.rp = TD(undef, length(ref.rp))
    sol.rd = TD(undef, length(ref.rd))
    sol.rp_old = TD(undef, length(ref.rp_old))
    sol.Ax_plus_By = TD(undef, length(ref.Ax_plus_By))
    sol.wRIij = TD(undef, length(ref.wRIij))

    copyto!(sol.x_curr, ref.x_curr)
    copyto!(sol.xbar_curr, ref.xbar_curr)
    copyto!(sol.z_outer, ref.z_outer)
    copyto!(sol.z_curr, ref.z_curr)
    copyto!(sol.z_prev, ref.z_prev)
    copyto!(sol.l_curr, ref.l_curr)
    copyto!(sol.lz, ref.lz)
    copyto!(sol.rho, ref.rho)
    copyto!(sol.rp, ref.rp)
    copyto!(sol.rd, ref.rd)
    copyto!(sol.rp_old, ref.rp_old)
    copyto!(sol.Ax_plus_By, ref.Ax_plus_By)
    copyto!(sol.wRIij, ref.wRIij)
    return sol
end

abstract type AbstractUserIterationInformation end

mutable struct ComponentInformation <: AbstractUserIterationInformation
    err_pg::Float64
    err_qg::Float64
    err_vm::Float64
    err_real::Float64
    err_reactive::Float64
    err_rateA::Float64
    err_ramp::Float64
    num_rateA_viols::Int
    time_generators::Float64
    time_branches::Float64
    time_buses::Float64

    function ComponentInformation()
        user = new()
        fill!(user, 0)
        return user
    end
end

function Base.fill!(user::ComponentInformation, val)
    user.err_pg = val
    user.err_qg = val
    user.err_vm = val
    user.err_real = val
    user.err_reactive = val
    user.err_rateA = val
    user.err_ramp = val
    user.num_rateA_viols = val
    user.time_generators = val
    user.time_branches = val
    user.time_buses = val
    return
end

function Base.copy(ref::ComponentInformation)
    user = ComponentInformation()
    user.err_pg = ref.err_pg
    user.err_qg = ref.err_qg
    user.err_vm = ref.err_vm
    user.err_real = ref.err_real
    user.err_reactive = ref.err_reactive
    user.err_rateA = ref.err_rateA
    user.err_ramp = ref.err_ramp
    user.num_rateA_viols = ref.num_rateA_viols
    user.time_generators = ref.time_generators
    user.time_branches = ref.time_branches
    user.time_buses = ref.time_buses
    return user
end

mutable struct IterationInformation{U}
    status::Symbol
    inner::Int
    outer::Int
    cumul::Int
    objval::Float64
    primres::Float64
    dualres::Float64
    mismatch::Float64
    eps_pri::Float64
    norm_z_curr::Float64
    norm_z_prev::Float64
    time_x_update::Float64
    time_xbar_update::Float64
    time_z_update::Float64
    time_l_update::Float64
    time_lz_update::Float64
    time_projection::Float64
    time_overall::Float64

    user::AbstractUserIterationInformation

    function IterationInformation{U}() where {U <: AbstractUserIterationInformation}
        info = new()
        info.status = :NotSpecified
        info.user = U()
        fill!(info, 0)
        return info
    end
end


function Base.fill!(info::IterationInformation, val)
    info.inner = val
    info.outer = val
    info.cumul = val
    info.objval = val
    info.primres = val
    info.dualres = val
    info.mismatch = val
    info.eps_pri = val
    info.norm_z_curr = val
    info.norm_z_prev = val
    info.time_x_update = val
    info.time_xbar_update = val
    info.time_z_update = val
    info.time_l_update = val
    info.time_lz_update = val
    info.time_projection = val
    info.time_overall = val
    fill!(info.user, val)
end

function Base.copy(ref::IterationInformation{ComponentInformation})
    info = IterationInformation{ComponentInformation}()
    info.status = ref.status
    info.inner = ref.inner
    info.outer = ref.outer
    info.cumul = ref.cumul
    info.objval = ref.objval
    info.primres = ref.primres
    info.dualres = ref.dualres
    info.mismatch = ref.mismatch
    info.eps_pri = ref.eps_pri
    info.norm_z_curr = ref.norm_z_curr
    info.norm_z_prev = ref.norm_z_prev
    info.time_x_update = ref.time_x_update
    info.time_xbar_update = ref.time_xbar_update
    info.time_z_update = ref.time_z_update
    info.time_l_update = ref.time_l_update
    info.time_lz_update = ref.time_lz_update
    info.time_overall = ref.time_overall
    info.user = copy(ref.user)
    return info
end

abstract type AbstractOPFModel{T,TD,TI,TM} end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
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

    membuf::TM      # memory buffer for line kernel
    gen_membuf::TM  # memory buffer for generatork kernel

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    function Model{T,TD,TI,TM}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        return new{T,TD,TI,TM}()
    end

    function Model{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}; ramp_ratio=0.02) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
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
function Base.copy(ref::Model{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
    model = Model{T,TD,TI,TM}()

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