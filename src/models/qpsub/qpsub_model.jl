## T = Float64; TD = Array{Float64,1}; TI = Array{Int64,1}; TM = Array{Float64,2}

##############
"""
ParameterSQP

This contains the parameters used in SQP + ADMM algorithm.
"""
mutable struct ParameterSQP
    trust_rad::Float64
    function ParameterSQP()
        par = new()
        # par.trust_rad = 1.0
        return par
    end
end

##############
abstract type AbstractAdmmEnvSQP{T,TD,TI,TM} end
"""
    AdmmEnv{T,TD,TI,TM}

This structure carries data source and ADMM parameters required.
"""
mutable struct AdmmEnvSQP{T,TD,TI,TM} <: AbstractAdmmEnvSQP{T,TD,TI,TM}
    #grid info
    case::String
    data::OPFData  
    params::ParameterSQP
    #ParameterSQP

    function AdmmEnvSQP{T,TD,TI,TM}(
        case::String; case_format="matpower", verbose::Int=1
    )where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}

    env = new{T,TD,TI,TM}()
    env.case = case
    env.data = opf_loaddata(env.case;
    VI=TI, VD=TD, case_format=case_format,verbose=verbose) 
    env.params = ParameterSQP()
      return env
    end
end


##############
abstract type AbstractSolutionSQP{T,TD} end
"""
    SolutionACOPF{T,TD}

This contains the current ACOPF solution from SQP solver 
"""
mutable struct SolutionACOPF{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_acopf
    pg::TD
    qg::TD
    #curr_pi_acopf
    function SolutionACOPF{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #pg
            TD(undef,ngen), #qg
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionACOPF, val)
    fill!(sol.pg, val)
    fill!(sol.qg, val)
end

"""
    Coeff_SQP{T,TD}

This contains the other coefficients from SQP solver (e.g., bounds on d_pg) 
"""
mutable struct Coeff_SQP{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_acopf
    dpg_min::TD
    dpg_max::TD
    dqg_max::TD
    dqg_min::TD
    #curr_pi_acopf
    function Coeff_SQP{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #dpg_min
            TD(undef,ngen), #dpg_max
            TD(undef,ngen), #dqg_min
            TD(undef,ngen), #dqg_max
        )
        fill!(sol, 0.0, 1.0)
        return sol
    end
end

function Base.fill!(sol::Coeff_SQP, valmin, valmax)
    fill!(sol.dpg_min, valmin)
    fill!(sol.dpg_max, valmax)
    fill!(sol.dpg_min, valmin)
    fill!(sol.dpg_max, valmax)
end



"""
    SolutionQP_gen{T,TD}

This contains the solutions of ALL generator QP subproblem 
"""
mutable struct SolutionQP_gen{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_genQP
    pg::TD
    qg::TD

    function SolutionQP_gen{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #pg
            TD(undef,ngen), #qg
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionQP_gen, val)
    fill!(sol.pg, val)
    fill!(sol.qg, val)
end

"""
    SolutionQP_bus{T,TD}

This contains the solutions of ALL bus QP subproblem 
"""
mutable struct SolutionQP_bus{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_busQP
    #note: each bus may contain multiple generators but each generator is only copied once
    pg::TD
    qg::TD
    function SolutionQP_bus{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #pg
            TD(undef,ngen), #qg
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionQP_bus, val)
    fill!(sol.pg, val)
    fill!(sol.qg, val)
end

"""
    SolutionQP_br{T,TD}

This contains the solutions of ALL branch QP subproblem 
"""
mutable struct SolutionQP_br{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_brQP
    
end

##############
"""
lam_rho_pi_gen

This contains the rho and lamb parameters used in genQP and busQP.
"""
mutable struct Lam_rho_pi_gen{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_genQP
    lam_pg::TD
    lam_qg::TD
    rho_pg::TD
    rho_qg::TD

    function Lam_rho_pi_gen{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #lam_pg
            TD(undef,ngen), #lam_qg
            TD(undef,ngen), #rho_pg
            TD(undef,ngen), #rho_qg
        )
        fill!(sol, 1.0)
        return sol
    end
end

function Base.fill!(sol::Lam_rho_pi_gen, val)
    fill!(sol.lam_pg, val)
    fill!(sol.lam_qg, val)
    fill!(sol.rho_pg, val)
    fill!(sol.rho_qg, val)
end



##############
abstract type AbstractOPFModelSQP{T,TD,TI,TM} end

"""
    Model{T,TD,TI}

This contains the parameters and solutions in all problem solving.
"""
mutable struct ModelSQP{T,TD,TI,TM} <: AbstractOPFModelSQP{T,TD,TI,TM}
    #SolutionQP
    #SolutionACOPF 
end
