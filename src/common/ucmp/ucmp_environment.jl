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

  function SolutionUC{T,TD}(ngen::Int, ncouple::Int, _t::Int, _len_horizon::Int) where {T,TD<:AbstractArray{T}}
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

"""
    UCAdmmEnv{T,TD,TI}

This structure carries everything required to run ADMM from a given solution.
"""
mutable struct UCAdmmEnv{T,TD,TI,TM} <: AbstractAdmmEnv{T,TD,TI,TM}
    env::AdmmEnv{T,TD,TI,TM}
    uc_specified::Bool
    uc_data::TM

    function UCAdmmEnv{T,TD,TI,TM}(
        case::String, rho_pq::Float64, rho_va::Float64;
        case_format="matpower",
        use_gpu=false, use_linelimit=false, use_twolevel=false, use_mpi=false, use_projection=false,
        gpu_no::Int=1, verbose::Int=1, tight_factor=1.0,
        horizon_length=1, load_prefix::String="", gen_prefix::String="", comm::MPI.Comm=MPI.COMM_WORLD
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        ucenv = new{T,TD,TI,TM}()
        ucenv.env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
                                        use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
                                        load_prefix=load_prefix, tight_factor=tight_factor, gpu_no=gpu_no, verbose=verbose)
        ucenv.uc_specified = false
        if !isempty(gen_prefix)
            ucenv.uc_data = readdlm(name,',',TI)
            @assert size(ucenv.uc_data,1) == length(ucenv.env.data.generators)
            ucenv.uc_specified = true
        end
        return ucenv
    end
end

mutable struct UCMPModel{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    solution::Vector{SolutionRamping{T,TD}} # ramp related solution
    uc_solution::Vector{SolutionUC{T, TD}}  # uc related solution

    nvar::Int                               # total number of variables
    len_horizon::Int                        # the length of a time horizon
    ramp_ratio::T                           # ramp ratio
    models::Vector{Model{T,TD,TI,TM}}       # a collection of time periods

    uc_membufs::Vector{TM}                  # memory buffer for uc variables in generator kernel

    function UCMPModel{T,TD,TI,TM}(env::UCAdmmEnv{T,TD,TI,TM};
        start_period=1, end_period=1, ramp_ratio=0.02) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}

        @assert env.load_specified == true
        @assert start_period >= 1 && start_period <= end_period && end_period <= size(env.load.pd,2)

        mod = new{T,TD,TI,TM}()
        num_periods = end_period - start_period + 1
        mod.len_horizon = num_periods

        mod.models = Vector{UCModel{T,TD,TI,TM}}(undef, num_periods)
        mod.models[1] = Model{T,TD,TI,TM}(env; ramp_ratio=ramp_ratio)
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

        # TODO: decide the amount of memory needed for each of uc_membufs
        mod.uc_membufs = Vector{TM}(undef, num_periods)
        for i in 1:num_periods
            mod.uc_membufs[i] = TM(undef, (1,mod.models[1].ngen))
            fill!(mod.uc_membufs[i], 0.0)
        end

        mod.solution = Vector{SolutionRamping{T,TD}}(undef, num_periods)
        mod.uc_solution = Vector{SolutionUC{T,TD}}(undef, num_periods)
        ngen = mod.models[1].ngen
        for i=1:num_periods
            mod.solution[i] = SolutionRamping{T,TD}(ngen, ngen, i, num_periods)
            mod.uc_solution[i] = SolutionUC{T,TD}(ngen, ngen, i, num_periods)
        end
        # init_solution!(mod, mod.solution, env.initial_rho_pq, env.initial_rho_va)
        # TODO: think about how to initialize uc solution

        # TODO: double check what nvar means here
        # mod.nvar = 0
        # if num_periods == 1
        #     mod.nvar = mod.models[1].nvar
        # else
        #     mod.nvar = mod.models[1].nvar + mod.models[1].ngen
        # end

        mod.info = IterationInformation{ComponentInformation}()

        return mod
    end
end
