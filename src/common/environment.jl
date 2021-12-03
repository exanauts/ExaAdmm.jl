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
    load_specified::Bool
    gpu_no::Int
    comm::MPI.Comm

    params::Parameters
#    model::AbstractOPFModel{T,TD,TI}
#    membuf::TM # was param

    function AdmmEnv{T,TD,TI,TM}(
        case::String, rho_pq::Float64, rho_va::Float64;
        case_format="matpower",
        use_gpu=false, use_linelimit=false, use_twolevel=false, use_mpi=false,
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
    z_outer::TD    # used only for the two-level formulation
    z_curr::TD     # used only for the two-level formulation
    z_prev::TD     # used only for the two-level formulation
    lz::TD         # used only for the two-level formulation
    Ax_plus_By::TD # used only for the two-level formulation
    objval::T
    overall_time::T
    max_viol_except_line::T
    max_line_viol_rateA::T
    cumul_iters::Int
    status::Symbol

    function SolutionOneLevel{T,TD}(nvar::Int) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            TD(undef, nvar),
            Inf,
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
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.lz, val)
    fill!(sol.Ax_plus_By, val)
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
    objval::T

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
            TD(undef, 2*nline),   # wRIij
            Inf,
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

abstract type AbstractUserIterationInformation end

mutable struct ComponentInformation <: AbstractUserIterationInformation
    err_pg::Float64
    err_qg::Float64
    err_vm::Float64
    err_real::Float64
    err_reactive::Float64
    err_rateA::Float64
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
    user.num_rateA_viols = val
    user.time_generators = val
    user.time_branches = val
    user.time_buses = val
    return
end

mutable struct IterationInformation{U}
    inner::Int
    outer::Int
    cumul::Int
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
    time_overall::Float64

    user::AbstractUserIterationInformation

    function IterationInformation{U}() where {U <: AbstractUserIterationInformation}
        info = new()
        fill!(info, 0)
        info.user = U()
        return info
    end
end


function Base.fill!(info::IterationInformation, val)
    info.inner = val
    info.outer = val
    info.cumul = val
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
    info.time_overall = val
end

abstract type AbstractOPFModel{T,TD,TI,TM} end

"""
    Model{T,TD,TI}

This contains the parameters specific to ACOPF model instance.
"""
mutable struct Model{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    solution::AbstractSolution{T,TD}

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

    membuf::TM

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    function Model{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}; ramp_ratio=0.2) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
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

        model.membuf = TM(undef, (31, model.nline))
        fill!(model.membuf, 0.0)

        return model
    end
end

