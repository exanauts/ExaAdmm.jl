"""
    solve_qpsub()
    
- main function solve qpsub problem 
- TODO: integrate with SQP (check coefficient validity, use_projection)
- TODO: clear lower level TODOs
"""


function solve_qpsub(case::String;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, use_linelimit=true, use_projection=false, tight_factor=1.0,
    outer_eps=2*1e-4, gpu_no=0, verbose=1
)
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
            use_projection=use_projection, tight_factor=tight_factor, gpu_no=gpu_no,
            storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
            verbose=verbose)
    mod = ModelQpsub{T,TD,TI,TM}(env)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    env.params.shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    # admm_two_level(env, mod)

    # testing ADMM progress 
    # admm_test(env,mod)

    return env, mod
end