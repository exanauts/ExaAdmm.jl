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
    storage_ratio::T
    droop::T
    initial_rho_pq::T
    initial_rho_va::T
    tight_factor::T
    horizon_length::Int
    use_gpu::Bool
    use_linelimit::Bool
    use_mpi::Bool
    use_projection::Bool
    load_specified::Bool
    gpu_no::Int
    comm::MPI.Comm

    params::Parameters
#    model::AbstractOPFModel{T,TD,TI}
#    membuf::TM # was param

    function AdmmEnv{T,TD,TI,TM}(
        data::OPFData, case::String, rho_pq::Float64, rho_va::Float64;
        case_format="matpower",
        use_gpu=false, use_linelimit=true, use_mpi=false, use_projection=false,
        gpu_no::Int=0, verbose::Int=1, tight_factor=1.0, droop=0.04, storage_ratio=0.0, storage_charge_max=1.0,
        horizon_length=1, load_prefix::String="", comm::MPI.Comm=MPI.COMM_WORLD
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        Random.seed!(0)

        env = new{T,TD,TI,TM}()
        env.case = case
        env.data = data
        env.storage_ratio = storage_ratio
        env.droop = droop
        env.initial_rho_pq = rho_pq
        env.initial_rho_va = rho_va
        env.tight_factor = tight_factor
        env.use_gpu = use_gpu
        env.use_linelimit = use_linelimit
        env.use_mpi = use_mpi
        env.use_projection = use_projection
        env.gpu_no = gpu_no
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

function AdmmEnv{T,TD,TI,TM}(
    case::String, rho_pq::Float64, rho_va::Float64; options...
) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
    data = opf_loaddata(case; options...)
                             # VI=TI, VD=TD, case_format=case_format, verbose=verbose)
    return AdmmEnv{T, TD, TI, TM}(data, case, rho_pq, rho_va; options...)
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
    Solution{T,TD}

This contains the solutions of ACOPF model instance, including the ADMM parameter rho.
"""
mutable struct Solution{T,TD} <: AbstractSolution{T,TD}
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

    function Solution{T,TD}(nvar::Int) where {T, TD<:AbstractArray{T}}
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


function Base.fill!(sol::Solution, val)
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

function Base.copy(ref::Solution{T,TD}) where {T,TD<:AbstractArray{T}}
    nvar = length(ref.u_curr)
    sol = Solution{T,TD}(nvar)

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
    auglag::Float64
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
    info.auglag = val
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
    info.auglag = ref.auglag
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

#=
mutable struct ComplementarityModel{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    grid::AbstractPowerGrid{T,TD,TI,TM}

    info::IterationInformation
    solution::AbstractSolution{T,TD}
    gen_solution::AbstractSolution{T,TD}

    n::Int
    nvar::Int
    nvar_padded::Int
    nline_padded::Int

    gen_start::Int
    line_start::Int

    pgmin_curr::TD   # taking ramping into account for rolling horizon
    pgmax_curr::TD   # taking ramping into account for rolling horizon

    membuf::TM       # memory ubuffer for line kernel

    function ComplementarityModel{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()
        model.grid = PowerGrid{T,TD,TI,TM}(env)

        # Additional variables for voltage stability: ngen
        #  - voltage magnitude for the bus attached to a generator
        #
        # Additional variables for frequency control: ngen
        #  - frequency deviation allocated for each generator
        #
        # Storage model variables: nstorage
        #  - charge - discharge amount
        #
        # Energy level will be included in the future along with multi-period implementation.
        #
        # If we include them all, the # of additional variables: 2*ngen + nstorage
        # Including (Pg,Qg), the total number of generator variables will be 4*ngen + nstorage.

        model.n = (env.use_linelimit == true) ? 6 : 4
        model.nline_padded = model.grid.nline
        model.nvar = 4*model.grid.ngen + model.grid.nstorage + 8*model.grid.nline
#        model.nvar = 3*model.grid.ngen + 8*model.grid.nline
        model.nvar_padded = model.nvar + 8*(model.nline_padded - model.grid.nline)
        model.gen_start = 1
        model.line_start = 4*model.grid.ngen + model.grid.nstorage + 1
#        model.line_start = 3*model.grid.ngen + 1

        model.solution = SolutionOneLevel{T,TD}(model.nvar_padded)
        init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va)
        model.gen_solution = EmptyGeneratorSolution{T,TD}()

        model.pgmin_curr = TD(undef, model.grid.ngen)
        model.pgmax_curr = TD(undef, model.grid.ngen)
        copyto!(model.pgmin_curr, model.grid.pgmin)
        copyto!(model.pgmax_curr, model.grid.pgmax)

        model.membuf = TM(undef, (31, model.grid.nline))
        fill!(model.membuf, 0.0)
        model.membuf[29,:] .= model.grid.rateA

        model.info = IterationInformation{ComponentInformation}()

        return model
    end
end
=#
