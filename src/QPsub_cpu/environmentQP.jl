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
        par.trust_rad = 1.0
        return par
    end
end

##############
abstract type AbstractAdmmEnvSQP{T,TD,TI,TM} end
"""
    AdmmEnv{T,TD,TI,TM}

This structure carries everything required to run ADMM from a current reference solution.
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

This contains the current ACOPF solution to generate QP subproblem including Lagrangian Multiplier PI + Radius Δ
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
    #curr_lambda_busQP
end

"""
    SolutionQP_br{T,TD}

This contains the solutions of ALL branch QP subproblem 
"""
mutable struct SolutionQP_br{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_brQP
    #curr_lambda_brQP
end

##############
"""
lam_rho_gen

This contains the rho and lamb parameters used in genQP and busQP.
"""
mutable struct lam_rho_gen{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_genQP
    lam_pg::TD
    lam_qg::TD
    rho_pg::TD
    rho_qg::TD

    function lam_rho_gen{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
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

function Base.fill!(sol::lam_rho_gen, val)
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
