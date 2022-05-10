## T = Float64; TD = Array{Float64,1}; TI = Array{Int64,1}; TM = Array{Float64,2}; TS = Array{Float64,3}

##############
# """
# ParameterSQP

# This contains the parameters used in SQP + ADMM algorithm.
# """
# mutable struct ParameterSQP
#     trust_rad::Float64
#     function ParameterSQP()
#         par = new()
#         # par.trust_rad = 1.0
#         return par
#     end
# end

##############
# abstract type AbstractAdmmEnvSQP{T,TD,TI,TM} end
# """
#     AdmmEnv{T,TD,TI,TM}

# This structure carries data source and ADMM parameters required.
# """
# mutable struct AdmmEnvSQP{T,TD,TI,TM} <: AbstractAdmmEnvSQP{T,TD,TI,TM}
#     #grid info
#     case::String
#     data::OPFData  
#     params::ParameterSQP
#     #ParameterSQP

#     function AdmmEnvSQP{T,TD,TI,TM}(
#         case::String; case_format="matpower", verbose::Int=1
#     )where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}

#     env = new{T,TD,TI,TM}()
#     env.case = case
#     env.data = opf_loaddata(env.case;
#     VI=TI, VD=TD, case_format=case_format,verbose=verbose) 
#     env.params = ParameterSQP()
#       return env
#     end
# end


############## append matrix type to AbstractSolution{T,TD}
abstract type AbstractSolutionSQP{T,TD,TM,TS} end
"""
    SolutionACOPF{T,TD}

This contains the current ACOPF solution from SQP solver 
"""
mutable struct SolutionACOPF{T,TD} <: AbstractSolution{T,TD}
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

This contains the other coefficients from SQP solver (e.g., bounds on d_i, constraint coefficient, Hessian and linear term) 
"""
mutable struct Coeff_SQP{T,TD,TM,TS} <: AbstractSolutionSQP{T,TD,TM,TS}
    #curr_read from SQP
    #gen
    dpg_min::TD
    dpg_max::TD
    dqg_max::TD
    dqg_min::TD
    Hpg::TM #hessian term for all pg; f = 1/2dHd + hd  
    hpg::TD #linear term for all pg 

    #line 6 variables in each line (i,j) [w_i w_j wRij wIij theta_i theta_j]
    Hpij::TS #hessian term for all lines; f = 1/2dHd + hd   
    hpij::TM #linear for all lines
    dwRij_min::TD
    dwRij_max::TD
    dwIij_min::TD
    dwIij_max::TD

    #bus (share with line [w_i theta_i])
    dwi_min::TD
    dwi_max::TD
    dthetai_min::TD
    dthetai_max::TD


    ## coefficient for linear equality 



    function Coeff_SQP{T,TD,TM,TS}(ngen::Int64, nline::Int64, nbus::Int64) where {T, TD<:AbstractArray{T}, TM<:AbstractArray{T,2}, TS<:AbstractArray{T,3}}
        sol = new{T,TD,TM,TS}(
            #gen 
            TD(undef,ngen), #dpg_min
            TD(undef,ngen), #dpg_max
            TD(undef,ngen), #dqg_min
            TD(undef,ngen), #dqg_max
            TM(undef,ngen,ngen), #Hpg
            TD(undef,ngen), #hpg

            #line
            TS(undef,6,6,nline), #Hpij
            TM(undef,6,nline), #hpij
            TD(undef,nline), #dwRij_min
            TD(undef,nline), #dwRij_max
            TD(undef,nline), #dwIij_min
            TD(undef,nline), #dwIij_max

            #bus
            TD(undef,nbus), #dwi_min
            TD(undef,nbus), #dwi_max
            TD(undef,nbus), #dthetai_min
            TD(undef,nbus) #dthetai_max
        )
        fill!(sol, 0.0, 1.0)
        return sol
    end
end

function Base.fill!(sol::Coeff_SQP, valmin, valmax)
    #gen 
    fill!(sol.dpg_min, valmin) #dpg_min
    fill!(sol.dpg_max, valmax) #dpg_max
    fill!(sol.dqg_min, valmin) #dqg_min
    fill!(sol.dqg_max, valmax) #dqg_max
    fill!(sol.Hpg,valmax) #Hpg
    fill!(sol.hpg,valmax) #hpg

    #line
    fill!(sol.Hpij,valmax) #Hpij
    fill!(sol.hpij,valmax) #hpij 
    fill!(sol.dwRij_min,valmin) #dwRij_min
    fill!(sol.dwRij_max,valmax) #dwRij_max
    fill!(sol.dwIij_min,valmin) #dwIij_min
    fill!(sol.dwIij_max,valmax) #dwIij_max

    #bus
    fill!(sol.dwi_min,valmin) #dwi_min
    fill!(sol.dwi_max,valmax) #dwi_max
    fill!(sol.dthetai_min,valmin) #dthetai_min
    fill!(sol.dthetai_max,valmax) #dthetai_max
end



"""
    SolutionQP_gen{T,TD}

This contains the solutions of ALL generator QP subproblem 
"""
mutable struct SolutionQP_gen{T,TD} <: AbstractSolution{T,TD}
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
mutable struct SolutionQP_bus{T,TD} <: AbstractSolution{T,TD}
    #curr_sol_busQP
    #note: each bus may contain multiple generators but each generator is only copied once
    dpg_c::TD
    dqg_c::TD
    dwi::TD
    dthetai::TD 
    # dwRij_i_c::TM
    # dwRji_j_c::TM
    # dwIij_i_c::TM
    # dwIji_j_c::TM

    function SolutionQP_bus{T,TD}(ngen::Int64,nbus::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,ngen), #dpg_c
            TD(undef,ngen), #dqg_c
            TD(undef,nbus), #dwi
            TD(undef,nbus) #dtheta
        )
        fill!(sol, 0.0)
        return sol
    end
end

function Base.fill!(sol::SolutionQP_bus, val)
    fill!(sol.dpg_c, val) #dpg_c
    fill!(sol.dqg_c, val) #dqg_c
    fill!(sol.dwi, val) #dwi
    fill!(sol.dthetai, val) #dthetai
end

"""
    SolutionQP_br{T,TD}

This contains the solutions of ALL branch QP subproblem 
"""
mutable struct SolutionQP_br{T,TD} <: AbstractSolution{T,TD}
    #curr_sol_brQP
    dwRij::TD
    dwIij::TD
    
    function SolutionQP_br{T,TD}(nline::Int64) where {T, TD<:AbstractArray{T}}
        sol = new{T,TD}(
            TD(undef,nline), #dwRij
            TD(undef,nline), #dwIij
        )
        fill!(sol, 0.0)
        return sol
    end   
end

function Base.fill!(sol::SolutionQP_br, val)
    fill!(sol.dwRij, val) #dpg_c
    fill!(sol.dwIij, val) #dqg_c
end

##############
"""
lam_rho_pi_gen

This contains the rho and lamb parameters used in genQP and busQP.
"""
mutable struct Lam_rho_pi_gen{T,TD} <: AbstractSolution{T,TD}
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
abstract type AbstractOPFModelSQP{T,TD,TI,TM,TS} end

"""
    Model{T,TD,TI}

This contains the parameters and solutions in all problem solving.
"""
mutable struct ModelQpsub{T,TD,TI,TM,TS} <: AbstractOPFModelSQP{T,TD,TI,TM,TS}
    #params
    ngen::Int64
    nline::Int64
    nbus::Int64

    c2::TD
    c1::TD
    c0::TD  

    #SolutionACOPF 
    acopf_sol::AbstractSolution{T,TD}

    #SolutionQP
    gen_qp::AbstractSolution{T,TD}
    bus_qp::AbstractSolution{T,TD}
    br_qp::AbstractSolution{T,TD}

    #lamda_rho_pi
    lam_rho_pi_gen::AbstractSolution{T,TD}

    #coeff from SQP
    coeff_sqp::AbstractSolutionSQP{T,TD,TM,TS}
    
    


    function ModelQpsub{T,TD,TI,TM,TS}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2},TS<:AbstractArray{T,3}}
        return new{T,TD,TI,TM,TS}()
    end

    # function ModelQpsub{T,TD,TI,TM}(env::AdmmEnvSQP{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}} #new environ
    function ModelQpsub{T,TD,TI,TM,TS}(env::AdmmEnv{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2},TS<:AbstractArray{T,3}} #old environ
        model = new{T,TD,TI,TM,TS}()

        #params
        model.ngen = length(env.data.generators)
        model.nline = length(env.data.lines)
        model.nbus = length(env.data.buses)

        model.c0 = Float64[env.data.generators[g].coeff[3] for g in 1:model.ngen]
        model.c1 = Float64[env.data.generators[g].coeff[2] for g in 1:model.ngen]
        model.c2 = Float64[env.data.generators[g].coeff[1] for g in 1:model.ngen]

        #solution acopf
        model.acopf_sol = SolutionACOPF{T,TD}(model.ngen)
        

        #solution qp
        model.gen_qp = SolutionQP_gen{T,TD}(model.ngen)
        model.bus_qp = SolutionQP_bus{T,TD}(model.ngen, model.nbus)
        model.br_qp = SolutionQP_br{T,TD}(model.nline)

        #lambda_rho_pi
        model.lam_rho_pi_gen = Lam_rho_pi_gen{T,TD}(model.ngen)

        #coeff from SQP 
        model.coeff_sqp = Coeff_SQP{T,TD,TM,TS}(model.ngen, model.nline, model.nbus)

        return model 
    end
end
