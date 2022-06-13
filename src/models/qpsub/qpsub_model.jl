"""
    Model{T,TD,TI,TM}

This contains the parameters specific to ACOPF model instance.
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
    
    
    qpsub_membuf::TM #memory buffer for qpsub
    
    
    
    #from SQP
    Hs::TM  # Hessian information for all lines 6*nline x 6: |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
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




    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

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

        if env.params.obj_scale != 1.0
            model.grid_data.c2 .*= env.params.obj_scale
            model.grid_data.c1 .*= env.params.obj_scale
            model.grid_data.c0 .*= env.params.obj_scale
        end

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.grid_data.ngen + 8*model.grid_data.nline
        model.nvar_u_padded = model.nvar_u + 8*(model.nline_padded - model.grid_data.nline)
        model.nvar_v = 2*model.grid_data.ngen + 4*model.grid_data.nline + 2*model.grid_data.nbus
        model.bus_start = 2*model.grid_data.ngen + 4*model.grid_data.nline + 1
        if env.use_twolevel
            model.nvar = model.nvar_u + model.nvar_v
            model.nvar_padded = model.nvar_u_padded + model.nvar_v
        end

        # Memory space is allocated based on the padded size.
        model.solution = ifelse(env.use_twolevel,
            SolutionTwoLevel{T,TD}(model.nvar_padded, model.nvar_v, model.nline_padded),
            SolutionOneLevel{T,TD}(model.nvar_padded))
        init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va)
        model.gen_solution = EmptyGeneratorSolution{T,TD}()
        
        #old memory buffer used in the auglag_linelimit with Tron 
        model.membuf = TM(undef, (31, model.grid_data.nline))
        fill!(model.membuf, 0.0)
        model.membuf[29,:] .= model.grid_data.rateA


        model.info = IterationInformation{ComponentInformation}()


        #new memory buffer used in the new auglag_Ab with Tron
        model.qpsub_membuf = TM(undef, (5,model.grid_data.nline))
        fill!(model.qpsub_membuf, 0.0)

        #new SQP parameters
        model.Hs = TM(undef,(6*model.grid_data.nline,6))
        fill!(model.Hs, 0.0)
        
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
    model.qpsub_membuf = copy(ref.qpsub_membuf) #eewly added

    model.nvar_u = ref.nvar_u
    model.nvar_v = ref.nvar_v
    model.bus_start = ref.bus_start

    model.nline_padded = ref.nline_padded
    model.nvar_padded = ref.nvar_padded
    model.nvar_u_padded = ref.nvar_u_padded

    return model
end