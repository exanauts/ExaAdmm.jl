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
    fill!(sol.dqg_min, valmin)
    fill!(sol.dqg_max, valmax)
end



"""
    SolutionQP_gen{T,TD}

This contains the solutions of ALL generator QP subproblem 
"""
mutable struct SolutionQP_gen{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_genQP
    dpg::TD
    dqg::TD

    function SolutionQP_gen{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #dpg
            TD(undef,ngen), #dqg
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionQP_gen, val)
    fill!(sol.dpg, val)
    fill!(sol.dqg, val)
end

"""
    SolutionQP_bus{T,TD}

This contains the solutions of ALL bus QP subproblem 
"""
mutable struct SolutionQP_bus{T,TD} <: AbstractSolutionSQP{T,TD}
    #curr_sol_busQP
    #note: each bus may contain multiple generators but each generator is only copied once
    dpg::TD
    dqg::TD
    function SolutionQP_bus{T,TD}(ngen::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #dpg
            TD(undef,ngen), #dqg
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionQP_bus, val)
    fill!(sol.dpg, val)
    fill!(sol.dqg, val)
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
mutable struct ModelQpsub{T,TD,TI,TM} <: AbstractOPFModelSQP{T,TD,TI,TM}
    #SolutionACOPF 
    acopf_sol::AbstractSolutionSQP{T,TD}
    #SolutionQP
    gen_qp::AbstractSolutionSQP{T,TD}
    bus_qp::AbstractSolutionSQP{T,TD}

    #lamda_rho_pi
    lam_rho_pi_gen::AbstractSolutionSQP{T,TD}

    #coeff from SQP
    coeff_sqp::AbstractSolutionSQP{T,TD}
    
    ngen::Int64
    nline::Int64
    nbus::Int64

    c2::TD
    c1::TD
    c0::TD 


    function ModelQpsub{T,TD,TI,TM}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        return new{T,TD,TI,TM}()
    end

    # function ModelQpsub{T,TD,TI,TM}(env::AdmmEnvSQP{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}} #new environ
    function ModelQpsub{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}} #old environ
        model = new{T,TD,TI,TM}()
        model.ngen = length(env.data.generators)
        model.nline = length(env.data.lines)
        model.nbus = length(env.data.buses)

        model.acopf_sol = SolutionACOPF{T,TD}(model.ngen)
        model.gen_qp = SolutionQP_gen{T,TD}(model.ngen)
        model.bus_qp = SolutionQP_bus{T,TD}(model.ngen)
        model.lam_rho_pi_gen = Lam_rho_pi_gen{T,TD}(model.ngen)
        model.coeff_sqp = Coeff_SQP{T,TD}(model.ngen)

        model.c2=zeros(Float64,model.ngen)
        model.c1=zeros(Float64,model.ngen)
        model.c0=zeros(Float64,model.ngen)
    for i = 1:model.ngen
        @inbounds model.c2[i]=env.data.generators[i].coeff[1]
        @inbounds model.c1[i]=env.data.generators[i].coeff[2]
        @inbounds model.c0[i]=env.data.generators[i].coeff[3]
    end

        return model 
    end
end
