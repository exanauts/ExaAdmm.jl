mutable struct UCParameters{TI}
    v0::TI # Generator initial statuses
    Tu::TI # Minimum up time
    Td::TI # Minimum down time
    Hu::TI # Up hours required at the beginning
    Hd::TI # Down hours required at the beginning

    function UCParameters{TI}(ngen::Int) where {TI <: AbstractArray}
        uc_params = new{TI}(
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen),
            TI(undef, ngen)
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
end

mutable struct SolutionUC{T,TD} <: AbstractSolution{T,TD}
    u_curr::TD
    v_curr::TD
    l_curr::TD
    rho::TD
    rd::TD
    rp::TD
    z_outer::TD
    z_curr::TD
    z_prev::TD
    lz::TD
    Ax_plus_By::TD
    s_curr::TD
    t::Int
    len_horizon::Int

    function SolutionUC{T,TD}(ngen::Int, _t::Int, _len_horizon::Int) where {T,TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, 3*ngen), # u_curr: [v, w, y]
            TD(undef, 3*ngen), # v_curr: [\bar{v}, \bar{w}, \bar{y}]
            TD(undef, 3*ngen), # l_curr: [λ_v, λ_w, λ_y]
            TD(undef, 3*ngen), # rho: [ρ_v, ρ_w, ρ_y]
            TD(undef, 3*ngen), # rd
            TD(undef, 3*ngen), # rp
            TD(undef, 3*ngen), # z_outer
            TD(undef, 3*ngen), # z_curr
            TD(undef, 3*ngen), # z_prev
            TD(undef, 3*ngen), # lz
            TD(undef, 3*ngen), # Ax_plus_By
            TD(undef, 6*ngen), # s_curr: [s^U, s^D, s^{p,UB}, s^{p,LB}, s^{q,UB}, s^{q,LB}]
            _t,                # t
            _len_horizon       # len_horizon
        )

        fill!(sol, 0)
        return sol
    end
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
    uc_solution::Vector{SolutionUC{T, TD}}  # uc related solution

    nvar::Int                               # total number of variables
    mpmodel::ModelMpacopf{T,TD,TI,TM}       # a collection of time periods

    uc_membufs::Vector{TM}                  # memory buffer for uc in generator kernel

    function UCMPModel{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}, gen_prefix::String;
        start_period=1, end_period=1, ramp_ratio=0.02) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}

        @assert env.load_specified == true
        @assert start_period >= 1 && start_period <= end_period && end_period <= size(env.load.pd,2)

        mod = new{T,TD,TI,TM}()
        num_periods = end_period - start_period + 1
        mod.mpmodel = ModelMpacopf{T,TD,TI,TM}(env, start_period=start_period, end_period=end_period, ramp_ratio=ramp_ratio)
        ngen = mod.mpmodel.models[1].grid_data.ngen
        mod.uc_params = UCParameters{TI}(ngen)

        # Load UC parameters
        v0s = readdlm(gen_prefix*".v0")
        Tus = readdlm(gen_prefix*".tu")
        Tds = readdlm(gen_prefix*".td")
        Hus = readdlm(gen_prefix*".hu")
        Hds = readdlm(gen_prefix*".hd")

        @assert ngen == length(v0s) == length(Tus) == length(Tds) == length(Hus) == length(Hds)

        copyto!(mod.uc_params.v0, v0s)
        copyto!(mod.uc_params.Tu, Tus)
        copyto!(mod.uc_params.Td, Tds)
        copyto!(mod.uc_params.Hu, Hus)
        copyto!(mod.uc_params.Hd, Hds)

        for submod in mod.mpmodel.models
            submod.gen_membuf = TM(undef, (43, ngen))
            fill!(submod.gen_membuf, 0.0)
        end

        # TODO: decide the amount of memory needed for each of uc_membufs
        mod.uc_membufs = Vector{TM}(undef, num_periods)
        for i in 1:num_periods
            mod.uc_membufs[i] = TM(undef, (1,ngen)) # TODO: replace 1 with the actual amount of memory needed
            fill!(mod.uc_membufs[i], 0.0)
        end

        mod.uc_solution = Vector{SolutionUC{T,TD}}(undef, num_periods)
        for i=1:num_periods
            mod.uc_solution[i] = SolutionUC{T,TD}(ngen, i, num_periods)
        end
        # init_solution!(mod, mod.uc_solution, env.initial_rho_pq, env.initial_rho_va) # TODO: implement init_solution! for uc model

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
