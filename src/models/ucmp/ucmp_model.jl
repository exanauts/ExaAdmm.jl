mutable struct UCParameters{TI,TD}
    v0::TI      # Generator initial statuses
    Tu::TI      # Minimum up time
    Td::TI      # Minimum down time
    Hu::TI      # Up hours required at the beginning
    Hd::TI      # Down hours required at the beginning
    con::TD     # Cost of switching on
    coff::TD    # Cost of switching off

    function UCParameters{TI,TD}(ngen::Int) where {TI <: AbstractArray, TD <: AbstractArray}
        uc_params = new{TI,TD}(
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen),
            TD(undef, ngen),
            TD(undef, ngen)
        )

        fill!(uc_params, 0)
        return uc_params
    end
end

function Base.fill!(uc_params::UCParameters, val)
    fill!(uc_params.v0, val)
    fill!(uc_params.Tu, val)
    fill!(uc_params.Td, val)
    fill!(uc_params.Hu, val)
    fill!(uc_params.Hd, val)
    fill!(uc_params.con, val)
    fill!(uc_params.coff, val)
end

# This is the solution struct for UC variables (v, w, y)
mutable struct SolutionUC{T,TM} <: AbstractSolution{T,TM}
    u_curr::TM
    v_curr::TM
    l_curr::TM
    rho::TM
    rd::TM
    rp::TM
    z_outer::TM
    z_curr::TM
    z_prev::TM
    lz::TM
    Ax_plus_By::TM
    s_curr::TM
    len_horizon::Int

    function SolutionUC{T,TM}(
        ngen::Int, 
        _len_horizon::Int
    ) where {T,TM<:AbstractArray{T, 2}}
        sol = new{T,TM}(
            TM(undef, ngen, 3*_len_horizon), # u_curr: [v, w, y]
            TM(undef, ngen, 3*_len_horizon), # v_curr: [\bar{v}, \bar{w}, \bar{y}]
            TM(undef, ngen, 7*_len_horizon), # l_curr: [λ_v, λ_w, λ_y, λ_pu, λ_pl, λ_qu, λ_ql]
            TM(undef, ngen, 7*_len_horizon), # rho: [ρ_v, ρ_w, ρ_y, ρ_pu, ρ_pl, ρ_qu, ρ_ql]
            TM(undef, ngen, 3*_len_horizon), # rd
            TM(undef, ngen, 3*_len_horizon), # rp
            TM(undef, ngen, 3*_len_horizon), # z_outer
            TM(undef, ngen, 3*_len_horizon), # z_curr
            TM(undef, ngen, 3*_len_horizon), # z_prev
            TM(undef, ngen, 3*_len_horizon), # lz
            TM(undef, ngen, 3*_len_horizon), # Ax_plus_By
            TM(undef, ngen, 4*_len_horizon), # s_curr: [s^{p,UB}, s^{p,LB}, s^{q,UB}, s^{q,LB}]
            _len_horizon       # len_horizon
        )

        fill!(sol, 0)
        return sol
    end
end

# The ramping solution records solutions of previous-time p and v.
# Not actually needed for t = 1.
function UCSolutionRamping(T, TD, ngen::Int, _t::Int, _len_horizon::Int)
    sol = SolutionRamping{T,TD}(
        TD(undef, 2*ngen),  # u_curr: [\hat{p}, \hat{v}]
        TD(undef, 2*ngen),  # v_curr: NOT USED, will couple with bus vars at previous time
        TD(undef, 4*ngen),  # l_curr: [λ_{\hat{p}}, λ_{\hat{v}}, λ_{su}, λ_{sd}]
        TD(undef, 4*ngen),  # rho: [ρ_{\hat{p}}, ρ_{\hat{v}}, ρ_{su}, ρ_{sd}]
        TD(undef, 2*ngen),  # rd
        TD(undef, 2*ngen),  # rp
        TD(undef, 2*ngen),  # z_outer
        TD(undef, 2*ngen),  # z_curr
        TD(undef, 2*ngen),  # z_prev
        TD(undef, 2*ngen),  # lz
        TD(undef, 2*ngen),  # Ax_plus_By
        TD(undef, 2*ngen),  # s_curr: [s^U, s^D]
        _t,                 # t
        _len_horizon        # len_horizon
    )
    return sol
end

function Base.fill!(sol::SolutionUC, val)
    fill!(sol.u_curr, val)
    fill!(sol.v_curr, val)
    fill!(sol.l_curr, val)
    fill!(sol.rho, val)
    fill!(sol.rd, val)
    fill!(sol.rp, val)
    fill!(sol.z_outer, val)
    fill!(sol.z_curr, val)
    fill!(sol.z_prev, val)
    fill!(sol.lz, val)
    fill!(sol.Ax_plus_By, val)
    fill!(sol.s_curr, val)
end

mutable struct UCMPModel{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    uc_params::UCParameters{TI}             # uc related parameters
    uc_solution::SolutionUC{T, TM}          # uc related solution

    nvar::Int                               # total number of variables
    mpmodel::ModelMpacopf{T,TD,TI,TM}       # a collection of time periods

    # uc_membufs::Vector{TM}                  # memory buffer for uc in generator kernel
    uc_membuf::TM                           # memory buffer for uc in generator kernel

    function UCMPModel{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}, gen_prefix::String;
        start_period=1, end_period=1, ramp_ratio=0.02) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}

        @assert env.load_specified == true
        @assert start_period >= 1 && start_period <= end_period && end_period <= size(env.load.pd,2)

        mod = new{T,TD,TI,TM}()
        num_periods = end_period - start_period + 1
        mod.mpmodel = ModelMpacopf{T,TD,TI,TM}(env, start_period=start_period, end_period=end_period, ramp_ratio=ramp_ratio)
        ngen = mod.mpmodel.models[1].grid_data.ngen
        mod.uc_params = UCParameters{TI,TD}(ngen)

        # Resize ramping solution in ModelMpacopf
        for i=1:num_periods
            mod.mpmodel.solution[i] = UCSolutionRamping(T, TD, ngen, i, num_periods)
        end

        # Load UC parameters
        v0s = readdlm(gen_prefix*".v0")
        Tus = readdlm(gen_prefix*".tu")
        Tds = readdlm(gen_prefix*".td")
        Hus = readdlm(gen_prefix*".hu")
        Hds = readdlm(gen_prefix*".hd")
        cons = readdlm(gen_prefix*".con")
        coffs = readdlm(gen_prefix*".coff")

        @assert ngen == length(v0s) == length(Tus) == length(Tds) == length(Hus) == length(Hds) == length(cons) == length(coffs)

        copyto!(mod.uc_params.v0, v0s)
        copyto!(mod.uc_params.Tu, Tus)
        copyto!(mod.uc_params.Td, Tds)
        copyto!(mod.uc_params.Hu, Hus)
        copyto!(mod.uc_params.Hd, Hds)
        copyto!(mod.uc_params.con, cons)
        copyto!(mod.uc_params.coff, coffs)

        for submod in mod.mpmodel.models
            submod.gen_membuf = TM(undef, (43, ngen))
            fill!(submod.gen_membuf, 0.0)
        end

        # TODO: decide the amount of memory needed for each of uc_membufs
        # mod.uc_membufs = Vector{TM}(undef, num_periods)
        # for i in 1:num_periods
        #     mod.uc_membufs[i] = TM(undef, (1,ngen)) # TODO: replace 1 with the actual amount of memory needed
        #     fill!(mod.uc_membufs[i], 0.0)
        # end
        mod.uc_membuf = TM(undef, (ngen, 3*num_periods))
        fill!(mod.uc_membuf, 0.0)

        mod.uc_solution = SolutionUC{T,TM}(ngen, num_periods)
        init_solution!(mod, mod.uc_solution, env.initial_rho_pq, env.initial_rho_va)

        # TODO: double check what nvar means here
        mod.nvar = 0
        if num_periods == 1
            mod.nvar = mod.mpmodel.models[1].nvar
        else
            mod.nvar = mod.mpmodel.models[1].nvar + ngen
        end

        mod.info = IterationInformation{ComponentInformation}()

        return mod
    end
end
