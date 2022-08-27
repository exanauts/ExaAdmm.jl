"""
    solve_qpsub()
    
- main function solve qpsub problem 
- TODO: integrate with SQP (check coefficient validity, use_projection)
- TODO: clear lower level TODOs
"""


# function solve_qpsub(case::String;
#     case_format="matpower",
#     outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
#     obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
#     use_gpu=false, use_linelimit=true, use_projection=false, tight_factor=1.0,
#     outer_eps=2*1e-4, gpu_no=0, verbose=1
# )
#     T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
#     if use_gpu
#         CUDA.device!(gpu_no)
#         TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
#     end

#     env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
#             use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
#             use_projection=use_projection, tight_factor=tight_factor, gpu_no=gpu_no,
#             storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
#             verbose=verbose)
#     mod = ModelQpsub{T,TD,TI,TM}(env)

#     env.params.scale = scale
#     env.params.obj_scale = obj_scale
#     env.params.outer_eps = outer_eps
#     env.params.outer_iterlim = outer_iterlim
#     env.params.inner_iterlim = inner_iterlim
#     env.params.shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

#     admm_two_level(env, mod)

#     return env, mod
# end

function solve_qpsub(case::String, Hs, LH_1h, RH_1h,
    LH_1i, RH_1i, LH_1j, RH_1j, LH_1k, RH_1k, ls, us, pgmax, pgmin, qgmax, qgmin, c1, c2, Pd, Qd, initial_beta;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, use_linelimit=true, use_projection=false, tight_factor=1.0,
    outer_eps=2*1e-4, gpu_no=0, verbose=1, onelevel = true
)
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, 
            # use_twolevel=false,
            use_projection=use_projection, tight_factor=tight_factor, gpu_no=gpu_no,
            storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
            verbose=verbose)
    mod = ModelQpsub{T,TD,TI,TM}(env)

    data = mod.grid_data

    # mod.Hs .= Hs
    # mod.LH_1h .= LH_1h
    # mod.RH_1h .= RH_1h
    # mod.LH_1i .= LH_1i
    # mod.RH_1i .= RH_1i
    # mod.LH_1j .= LH_1j
    # mod.RH_1j .= RH_1j
    # mod.LH_1k .= LH_1k
    # mod.RH_1k .= RH_1k
    # mod.ls .= ls
    # mod.us .= us

    # mod.qpsub_pgmax .= pgmax
    # mod.qpsub_pgmin .= pgmin
    # mod.qpsub_qgmax .= qgmax
    # mod.qpsub_qgmin .= qgmin
    
    # mod.qpsub_c1 .= c1
    # mod.qpsub_c2 .= c2
    # mod.qpsub_Pd .= Pd
    # mod.qpsub_Qd .= Qd

    mod.Hs = copy(Hs)
    mod.LH_1h = copy(LH_1h)
    mod.RH_1h = copy(RH_1h)
    mod.LH_1i = copy(LH_1i)
    mod.RH_1i = copy(RH_1i)
    mod.LH_1j = copy(LH_1j)
    mod.RH_1j = copy(RH_1j)
    mod.LH_1k = copy(LH_1k)
    mod.RH_1k = copy(RH_1k)
    mod.ls = copy(ls)
    mod.us = copy(us)
    
    mod.qpsub_pgmax = copy(pgmax)
    mod.qpsub_pgmin = copy(pgmin)
    mod.qpsub_qgmax = copy(qgmax)
    mod.qpsub_qgmin = copy(qgmin)

    mod.qpsub_c1 = copy(c1)
    mod.qpsub_c2 = copy(c2)
    mod.qpsub_Pd = copy(Pd)
    mod.qpsub_Qd = copy(Qd)

    

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    # env.params.shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)
    env.params.shmem_size = sizeof(Float64)*(16*mod.n+4*mod.n^2+178) + sizeof(Int)*(4*mod.n)

    env.params.initial_beta = initial_beta #use my initial beta 

    init_solution!(mod, mod.solution, env.initial_rho_pq, env.initial_rho_va)

    #debug check all values before using admm
    
    #onelevel or two level admm 
    if onelevel
    admm_one_level(env, mod)
    # println(mod.info.objval)
    else
    admm_two_level(env, mod)
    end
    # println(mod.info.objval)
    return env, mod
end