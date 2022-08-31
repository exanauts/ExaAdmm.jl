"""
    Model{T,TD,TI,TM}

This contains the parameters specific to QPSUB model instance.
"""


"""
    Solution Structure: 
- solution.u contain variables for generator and branch kernel
- solution.v contains variables for bus kernel
- Summary Table:

|  dimension     |   ngen  |  ngen  |  nline    |  nline  | nline     | nline    |  nline  |  nline   | nline      | nline       | 
|:--------------:|:-------:| :----: |:----:     |:----:   |:----:     |:----:    |:----:   |:----:    |:----:      |:----:       |
|structure for u |   pg    |  qg    |  p_ij     |  q_ij   |   p_ji    |  q_ji    | wi(ij)  |  wj(ji)  | thetai(ij) |  thetaj(ji) |
|structure for v |   pg(i) |  qg(i) |  p_ij(i)  |  q_ij(i)|   p_ji(j) |  q_ji(j) | wi      |  wj      | thetai     | thetaj      |

- structure for l and ρ is wrt all element of [x - xbar + z]  with same dimension  

- structure for sqp_line: 6*nline
    |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|

- Note: line has shared nodes => xbar contain duplications. 
    For example line(1,2) and line(2,3): w2 and theta2 exist twice in v.

- qpsub_membuf structure (dim = 5, used in auglag):
    - |λ_1h | λ_1i | λ_1j| λ_1k | ρ_{1h,1i,1j,1k}| 
    - For c(x) = 0, ALM = λ*c(x) + (ρ/2)c(x)^2
    - For 1j and 1k, introduce slack t_ij and t_ji (see internal branch structure for Exatron) 

"""
mutable struct ModelQpsub{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    info::IterationInformation
    solution::AbstractSolution{T,TD}

    # Used for multiple dispatch for multi-period case.
    gen_solution::AbstractSolution{T,TD}

    n::Int
    nvar::Int

    gen_start::Int
    line_start::Int

    pgmin_curr::TD   # taking ramping into account for rolling horizon
    pgmax_curr::TD   # taking ramping into account for rolling horizon

    grid_data::GridData{T,TD,TI,TM}

    membuf::TM      # memory buffer for line kernel
    gen_membuf::TM  # memory buffer for generator kernel
    
    v_prev::TD
   
    
    
    #qpsub_construct
    Hs::TM  # Hessian information for all lines 6*nline x 6: |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
    
    #QP coefficient for orignal branch kernel QP problem
    #only for testing for now 
    A_ipopt::TM #|w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)| 
    b_ipopt::TM #|w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
    
    
    LH_1h::TM  # nline * 4 : w_ijR w_ijI wi(ij) wj(ji)
    RH_1h::TD  # nline   
    LH_1i::TM  # nline * 4 : w_ijR w_ijI thetai(ij) thetaj(ji)
    RH_1i::TD  # nline    
    
    LH_1j::TM #nline * 2 : pij qij
    RH_1j::TD #nline 
    LH_1k::TM #nline * 2 : pji qji 
    RH_1k::TD #nline

    ls::TM # nline * 6
    us::TM # nline * 6 

    is_HS_sym::Array{Bool,1} #nline
    is_HS_PSD::Array{Bool,1} #nline 

    line_res::TM #4*nline

    qpsub_c1::TD #ngen
    qpsub_c2::TD #ngen

    qpsub_pgmax::TD #ngen
    qpsub_pgmin::TD #ngen
    qpsub_qgmax::TD #ngen
    qpsub_qgmin::TD #ngen

    qpsub_Pd::TD #nbus
    qpsub_Qd::TD #nbus



    #qpsub_solve
    qpsub_membuf::TM #memory buffer for qpsub 5*nline
    sqp_line::TM #6 * nline 
    
    #collect qpsub solution 
    dpg_sol::Array{Float64,1} #ngen
    dqg_sol::Array{Float64,1} #ngen

    dline_var::Array{Float64,2} #6*nline: w_ijR, w_ijI, w_i, w_j, theta_i, theta_j
    dline_fl::Array{Float64,2} #4*nline: p_ij, q_ij, p_ji, q_ji 

    dtheta_sol::Array{Float64,1} #nbus consensus with line_var
    dw_sol::Array{Float64,1} #nbus consensus with line_var

    #? moved to sqp model =>
    #SQP sol and param
    # pg_sol::TD #ngen
    # qg_sol::TD #ngen

    # line_var::TM #6*nline: w_ijR, w_ijI, w_i, w_j, theta_i, theta_j
    # line_fl::TM #4*nline: p_ij, q_ij, p_ji, q_ji 

    # theta_sol::TD #nbus consensus with line_var
    # w_sol::TD #nbus consensus with line_var

    # pft::TD #nbus support Pd balance
    # ptf::TD #nbus support Pd balance
    # pgb::TD #nbus support Pd balance

    # qft::TD #nbus support Qd balance
    # qtf::TD #nbus support Qd balance
    # qgb::TD #nbus support Qd balance

    #? moved to sqp model =>
    # eps_sqp::T
    # TR_sqp::TD #trust region radius for independent var 2*ngen + 4*nline: w_i, w_j, theta_i, theta_j
    # iter_lim_sqp::T
    # pen_merit::T #penalty for merit
    # FR_check::Bool #do FR or not
    # LF_check::Bool #do LF or not
    # SOC_check::Bool #do SOC or not 

    # bool_line::Array{Bool,2} #4* nline: 14h i j k violated violated or not for each line  
    # multi_line::TM #4*nline multiplier for 14h i j k 
     

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    # for integration
    dual_infeas::Array{Float64,1} #kkt error vector |pg |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
    lambda::TM #14h i j k 
    #multiplier

    # additional memory allocation for branch kernel (GPU)
    # NOTE: added by bowen 
    supY::TM #in gpu initialization 


    function ModelQpsub{T,TD,TI,TM}() where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        return new{T,TD,TI,TM}()
    end

    function ModelQpsub{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}; ramp_ratio=0.02) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()

        model.grid_data = GridData{T,TD,TI,TM}(env)

        model.n = (env.use_linelimit == true) ? 6 : 4 # branch kernel size (use linelimt or not)
        model.nline_padded = model.grid_data.nline

        # Memory space is padded for the lines as a multiple of # processes.
        if env.use_mpi
            nprocs = MPI.Comm_size(env.comm)
            model.nline_padded = nprocs * div(model.grid_data.nline, nprocs, RoundUp)
        end

        model.nvar = 2*model.grid_data.ngen + 8*model.grid_data.nline
        model.nvar_padded = model.nvar + 8*(model.nline_padded - model.grid_data.nline) #useless
        model.gen_start = 1 #location starting generator variables 
        model.line_start = 2*model.grid_data.ngen + 1 #location starting branch variables
        

        model.pgmin_curr = TD(undef, model.grid_data.ngen)
        model.pgmax_curr = TD(undef, model.grid_data.ngen)
        copyto!(model.pgmin_curr, model.grid_data.pgmin)
        copyto!(model.pgmax_curr, model.grid_data.pgmax)

        model.grid_data.ramp_rate = TD(undef, model.grid_data.ngen)
        model.grid_data.ramp_rate .= ramp_ratio.*model.grid_data.pgmax

        #scale the obj params with obj_scale
        if env.params.obj_scale != 1.0
            model.grid_data.c2 .*= env.params.obj_scale
            model.grid_data.c1 .*= env.params.obj_scale
            model.grid_data.c0 .*= env.params.obj_scale
            model.Hs .*=env.params.obj_scale
        end

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.grid_data.ngen + 8*model.grid_data.nline
        model.nvar_u_padded = model.nvar_u + 8*(model.nline_padded - model.grid_data.nline)
        model.nvar_v = 2*model.grid_data.ngen + 4*model.grid_data.nline + 2*model.grid_data.nbus
        model.bus_start = 2*model.grid_data.ngen + 4*model.grid_data.nline + 1
        # if env.use_twolevel
        #     model.nvar = model.nvar_u + model.nvar_v
        #     model.nvar_padded = model.nvar_u_padded + model.nvar_v
        # end

        # Memory space is allocated based on the padded size.
        model.solution = Solution{T,TD}(model.nvar_padded)
        # model.solution = ifelse(env.use_twolevel,
        #     SolutionTwoLevel{T,TD}(model.nvar_padded, model.nvar_v, model.nline_padded),
        #     Solution{T,TD}(model.nvar_padded))
        
        model.gen_solution = EmptyGeneratorSolution{T,TD}()
        
        #old memory buffer used in the auglag_linelimit with Tron 
        model.membuf = TM(undef, (31, model.grid_data.nline))
        fill!(model.membuf, 0.0)
        model.membuf[29,:] .= model.grid_data.rateA


        model.info = IterationInformation{ComponentInformation}()

        #new solution structure for tron (Hessian inherited from SQP: 6*nline)
        model.sqp_line = TM(undef, (6,model.grid_data.nline))
        fill!(model.sqp_line, 0.0)  

        #new v_prev for dual reshape
        model.v_prev = TD(undef, model.nvar)
        fill!(model.v_prev, 0.0) 

        #new memory buffer used in the new auglag_Ab with Tron
        model.qpsub_membuf = TM(undef, (5,model.grid_data.nline))
        fill!(model.qpsub_membuf, 0.0)

        #new qpsub parameters
        model.Hs = TM(undef,(6*model.grid_data.nline,6))
        fill!(model.Hs, 0.0)

        model.A_ipopt = TM(undef,(6*model.grid_data.nline,6))
        fill!(model.A_ipopt, 0.0)

        model.b_ipopt = TM(undef,(6,model.grid_data.nline))
        fill!(model.b_ipopt, 0.0)

        model.line_res = TM(undef,(4,model.grid_data.nline))
        fill!(model.line_res, 0.0)

        #1h
        model.LH_1h = TM(undef,(model.grid_data.nline,4))
        fill!(model.LH_1h, 0.0)

        model.RH_1h = TD(undef,model.grid_data.nline)
        fill!(model.RH_1h, 0.0)
        
        #1i
        model.LH_1i = TM(undef,(model.grid_data.nline,4))
        fill!(model.LH_1i, 0.0)

        model.RH_1i = TD(undef,model.grid_data.nline)
        fill!(model.RH_1i, 0.0)

        #1j 
        model.LH_1j = TM(undef,(model.grid_data.nline,2))
        fill!(model.LH_1j, 0.0)

        model.RH_1j = TD(undef,model.grid_data.nline)
        fill!(model.RH_1j, 0.0)

        #1k 
        model.LH_1k = TM(undef,(model.grid_data.nline,2))
        fill!(model.LH_1k, 0.0)

        model.RH_1k = TD(undef,model.grid_data.nline)
        fill!(model.RH_1k, 0.0)

        # l and u 
        model.ls = TM(undef,(model.grid_data.nline,6))
        fill!(model.ls, 0.0)

        model.us = TM(undef,(model.grid_data.nline,6))
        fill!(model.us, 0.0)

        model.is_HS_sym = Array{Bool,1}(undef, model.grid_data.nline)
        fill!(model.is_HS_sym, true)

        model.is_HS_PSD = Array{Bool,1}(undef, model.grid_data.nline)
        fill!(model.is_HS_PSD, true)

        model.qpsub_c1 = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_c1, 0.0)

        model.qpsub_c2 = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_c2, 0.0)

        model.qpsub_pgmax = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_pgmax, 0.0)

        model.qpsub_pgmin = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_pgmin, 0.0)

        model.qpsub_qgmax = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_qgmax, 0.0)

        model.qpsub_qgmin = TD(undef, model.grid_data.ngen)
        fill!(model.qpsub_qgmin, 0.0)

        model.qpsub_Pd = TD(undef, model.grid_data.nbus) 
        fill!(model.qpsub_Pd, 0.0)

        model.qpsub_Qd = TD(undef, model.grid_data.nbus)
        fill!(model.qpsub_Qd, 0.0)

        # SQP
        # model.pg_sol = TD(undef, model.grid_data.ngen)
        # fill!(model.pg_sol, 0.0)

        # model.qg_sol = TD(undef, model.grid_data.ngen)
        # fill!(model.qg_sol, 0.0)
        
        # model.line_var = TM(undef,(6, model.grid_data.nline))
        # fill!(model.line_var, 0.0)

        # model.line_fl = TM(undef,(4, model.grid_data.nline))
        # fill!(model.line_fl, 0.0)

        # model.theta_sol = TD(undef, model.grid_data.nbus)
        # fill!(model.theta_sol, 0.0)

        # model.w_sol = TD(undef, model.grid_data.nbus)
        # fill!(model.w_sol, 0.0)

        # model.pft = TD(undef, model.grid_data.nbus)
        # fill!(model.pft, 0.0)

        # model.ptf = TD(undef, model.grid_data.nbus)
        # fill!(model.ptf, 0.0)

        # model.pgb = TD(undef, model.grid_data.nbus)
        # fill!(model.pgb, 0.0)

        # model.qft = TD(undef, model.grid_data.nbus)
        # fill!(model.qft, 0.0)

        # model.qtf = TD(undef, model.grid_data.nbus)
        # fill!(model.qtf, 0.0)

        # model.qgb = TD(undef, model.grid_data.nbus)
        # fill!(model.qgb, 0.0)

        # model.TR_sqp = TD(undef, 2*model.grid_data.ngen + 4*model.grid_data.nline)
        # fill!(model.TR_sqp, TR)

        # model.eps_sqp = eps
        # model.iter_lim_sqp = iter_lim
        # model.pen_merit = 1.0 
        # model.FR_check = false
        # model.SOC_check = false
        # model.LF_check = false

        # model.bool_line = Array{Bool,2}(undef, (4, model.grid_data.nline))
        # fill!(model.bool_line, false)

        # model.multi_line = TM(undef, (4, model.grid_data.nline))
        # fill!(model.multi_line, 0.0)

        

        # qpsub solution
        model.dpg_sol = Array{Float64,1}(undef, model.grid_data.ngen)
        fill!(model.dpg_sol, 0.0)

        model.dqg_sol = Array{Float64,1}(undef, model.grid_data.ngen)
        fill!(model.dqg_sol, 0.0)
        
        model.dline_var = Array{Float64,2}(undef,(6, model.grid_data.nline))
        fill!(model.dline_var, 0.0)

        model.dline_fl = Array{Float64,2}(undef,(4, model.grid_data.nline))
        fill!(model.dline_fl, 0.0)

        model.dtheta_sol = Array{Float64,1}(undef, model.grid_data.nbus)
        fill!(model.dtheta_sol, 0.0)

        model.dw_sol = Array{Float64,1}(undef, model.grid_data.nbus)
        fill!(model.dw_sol, 0.0)

        # for SQP 
        model.dual_infeas = Array{Float64,1}(undef, model.grid_data.ngen + 6*model.grid_data.nline)
        fill!(model.dual_infeas, 1000.0)  

        model.lambda = TM(undef, (4, model.grid_data.nline))
        fill!(model.lambda, 0.0)

        #additional options
        model.supY = TM(undef, (4*model.grid_data.nline, 8)) #see supY def in eval_A_b_branch_kernel_gpu_qpsub
        fill!(model.supY, 0.0) 


        return model
    end
end

"""
This is to share power network data between models. Some fields that could be modified are deeply copied.
"""
function Base.copy(ref::ModelQpsub{T,TD,TI,TM}) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
    model = ModelQpsub{T,TD,TI,TM}()

    model.solution = copy(ref.solution)
    model.gen_solution = copy(ref.gen_solution)
    model.info = copy(ref.info)
    model.grid_data = copy(ref.grid_data)

    model.n = ref.n
    model.nvar = ref.nvar

    model.gen_start = ref.gen_start
    model.line_start = ref.line_start

    model.pgmin_curr = copy(ref.pgmin_curr)
    model.pgmax_curr = copy(ref.pgmax_curr)

    model.membuf = copy(ref.membuf)
    model.gen_membuf = copy(ref.gen_membuf)
    model.qpsub_membuf = copy(ref.qpsub_membuf) 

    model.nvar_u = ref.nvar_u
    model.nvar_v = ref.nvar_v
    model.bus_start = ref.bus_start

    model.nline_padded = ref.nline_padded
    model.nvar_padded = ref.nvar_padded
    model.nvar_u_padded = ref.nvar_u_padded

    model.Hs = copy(ref.Hs)
    model.LH_1h = copy(ref.LH_1h)
    model.RH_1h = copy(ref.RH_1h)

    model.LH_1i = copy(ref.LH_1i)
    model.RH_1i = copy(ref.RH_1i)

    model.LH_1j = copy(ref.LH_1j)
    model.RH_1j = copy(ref.RH_1j)

    model.LH_1k = copy(ref.LH_1k)
    model.RH_1k = copy(ref.RH_1k)

    model.ls = copy(ref.ls)
    model.us = copy(ref.us)

    model.line_res = copy(ref.line_res)

    model.sqp_line = copy(ref.sqp_line)

    # model.pg_sol = copy(ref.pg_sol)
    # model.qg_sol = copy(ref.qg_sol)
    # model.line_var = copy(ref.line_var)
    # model.line_fl = copy(ref.line_fl)
    # model.theta_sol = copy(ref.theta_sol)
    # model.w_sol = copy(ref.w_sol)
    # model.pft = copy(ref.pft)
    # model.ptf = copy(ref.ptf)
    # model.pgb = copy(ref.pgb)
    # model.qft = copy(ref.qft)
    # model.qtf = copy(ref.qtf)
    # model.qgb = copy(ref.qgb)

    # model.eps_sqp = copy(ref.eps_sqp)
    # model.iter_lim_sqp = copy(ref.iter_lim_sqp)
    # model.TR_sqp = copy(ref.TR_sqp)
    # model.pen_merit = ref.pen_merit
    # model.FR_check = ref.FR_check
    # model.SOC_check = ref.SOC_check
    # model.LF_check = ref.LF_check
    
    # model.bool_line = copy(ref.bool_line)
    # model.multi_line = copy(ref.multi_line) 

    model.dpg_sol = copy(ref.dpg_sol)
    model.dqg_sol = copy(ref.dqg_sol)
    model.dline_var = copy(ref.dline_var)
    model.dline_fl = copy(ref.dline_fl)
    model.dtheta_sol = copy(ref.dtheta_sol)
    model.dw_sol = copy(ref.dw_sol)

    model.is_HS_sym = copy(ref.is_HS_sym)
    model.is_HS_PSD = copy(ref.is_HS_PSD)

    model.qpsub_c1 = copy(ref.qpsub_c1)
    model.qpsub_c2 = copy(ref.qpsub_c2)

    model.qpsub_pgmax = copy(ref.qpsub_pgmax)
    model.qpsub_pgmin = copy(ref.qpsub_pgmin)
    model.qpsub_qgmax = copy(ref.qpsub_qgmax)
    model.qpsub_qgmin = copy(ref.qpsub_qgmin)

    model.qpsub_Pd = copy(ref.qpsub_Pd)
    model.qpsub_Qd = copy(ref.qpsub_Qd)

    # for SQP 
    model.dual_infeas = copy(ref.dual_infeas)
    model.lambda = copy(ref.lambda)

    #additional parameters 
    model.A_ipopt = copy(ref.A_ipopt)
    model.b_ipopt = copy(ref.b_ipopt)
    model.supY = copy(ref.supY)

    return model
end
