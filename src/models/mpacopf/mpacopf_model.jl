mutable struct SolutionRamping{T,TD} <: AbstractSolution{T,TD}
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

    function SolutionRamping{T,TD}(ngen::Int, ncouple::Int, _t::Int, _len_horizon::Int) where {T,TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef, ngen),    # u_curr
            TD(undef, ncouple), # v_curr
            TD(undef, ncouple), # l_curr
            TD(undef, ncouple), # rho
            TD(undef, ncouple), # rd
            TD(undef, ncouple), # rp
            TD(undef, ncouple), # z_outer
            TD(undef, ncouple), # z_curr
            TD(undef, ncouple), # z_prev
            TD(undef, ncouple), # lz
            TD(undef, ncouple), # Ax_plus_By
            TD(undef, ngen),    # s_curr
            _t,                 # t
            _len_horizon        # len_horizon
        )

        fill!(sol, 0)
        return sol
    end
  end

  function Base.fill!(sol::SolutionRamping, val)
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

mutable struct ModelMpacopf{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    solution::Vector{SolutionRamping{T,TD}} # ramp related solution

    nvar::Int                               # total number of variables
    len_horizon::Int                        # the length of a time horizon
    ramp_ratio::T                           # ramp ratio
    models::Vector{ModelAcopf{T,TD,TI,TM}}       # a collection of time periods

    function ModelMpacopf{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM};
        start_period=1, end_period=1, ramp_ratio=0.02) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}

        @assert env.load_specified == true
        @assert start_period >= 1 && start_period <= end_period && end_period <= size(env.load.pd,2)

        mod = new{T,TD,TI,TM}()
        num_periods = end_period - start_period + 1
        mod.len_horizon = num_periods

        mod.models = Vector{ModelAcopf{T,TD,TI,TM}}(undef, num_periods)
        mod.models[1] = ModelAcopf{T,TD,TI,TM}(env; ramp_ratio=ramp_ratio)
        mod.models[1].gen_membuf = TM(undef, (12,mod.models[1].ngen))
        fill!(mod.models[1].gen_membuf, 0.0)
        for i=2:num_periods
            mod.models[i] = copy(mod.models[1])
            mod.models[i].gen_membuf = TM(undef, (8,mod.models[i].ngen))
            fill!(mod.models[i].gen_membuf, 0.0)
        end
        for (i,t) in enumerate(start_period:end_period)
            mod.models[i].Pd .= env.load.pd[:,t]
            mod.models[i].Qd .= env.load.qd[:,t]
        end
        n = mod.models[1].n
        env.params.shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)
        env.params.gen_shmem_size = sizeof(Float64)*(14*3+3*3^2) + sizeof(Int)*(4*3)

        mod.solution = Vector{SolutionRamping{T,TD}}(undef, num_periods)
        ngen = mod.models[1].ngen
        for i=1:num_periods
            mod.solution[i] = SolutionRamping{T,TD}(ngen, ngen, i, num_periods)
        end
        init_solution!(mod, mod.solution, env.initial_rho_pq, env.initial_rho_va)

        mod.nvar = 0
        if num_periods == 1
            mod.nvar = mod.models[1].nvar
        else
            mod.nvar = mod.models[1].nvar + mod.models[1].ngen
        end

        mod.info = IterationInformation{ComponentInformation}()

        return mod
    end
end

mutable struct ModelMpacopfLoose{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    solution::Vector{SolutionRamping{T,TD}} # ramp related solution

    nvar::Int                               # total number of variables
    len_horizon::Int                        # the length of a time horizon
    ramp_ratio::T                           # ramp ratio
    models::Vector{ModelAcopf{T,TD,TI,TM}}       # a collection of time periods

    function ModelMpacopfLoose{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM};
        start_period=1, end_period=1, ramp_ratio=0.02) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}

        @assert env.load_specified == true
        @assert start_period >= 1 && start_period <= end_period && end_period <= size(env.load.pd,2)

        mod = new{T,TD,TI,TM}()
        num_periods = end_period - start_period + 1
        mod.len_horizon = num_periods

        mod.models = Vector{ModelAcopf{T,TD,TI,TM}}(undef, num_periods)
        mod.models[1] = Model{T,TD,TI,TM}(env; ramp_ratio=ramp_ratio)
        mod.models[1].gen_membuf = TM(undef, (8,mod.models[1].ngen))
        fill!(mod.models[1].gen_membuf, 0.0)
        for i=2:num_periods
            mod.models[i] = copy(mod.models[1])
            mod.models[i].gen_membuf = TM(undef, (8,mod.models[i].ngen))
            fill!(mod.models[i].gen_membuf, 0.0)
        end
        for (i,t) in enumerate(start_period:end_period)
            mod.models[i].Pd .= env.load.pd[:,t]
            mod.models[i].Qd .= env.load.qd[:,t]
        end

        mod.solution = Vector{SolutionRamping{T,TD}}(undef, num_periods)
        ngen = mod.models[1].ngen
        mod.nvar = 2*ngen*(num_periods-1)
        for i=1:num_periods
            mod.solution[i] = SolutionRamping{T,TD}(ngen, 2*ngen, i, num_periods)
            mod.models[i].gen_solution = mod.solution[i]
        end
        init_solution!(mod, mod.solution, env.initial_rho_pq, env.initial_rho_va)

        mod.info = IterationInformation{ComponentInformation}()

        return mod
    end
end