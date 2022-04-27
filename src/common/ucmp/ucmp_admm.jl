function solve_ucmp(case::String, load_prefix::String, gen_prefix::String;
    case_format="matpower", start_period=1, end_period=1,
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4,
    use_gpu=false, use_linelimit=true, tight_factor=0.99,
    outer_eps=2*1e-4, gpu_no=0, verbose=1, ramp_ratio=0.02, warm_start=true, multiperiod_tight=true)

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = UCAdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
            load_prefix=load_prefix, gen_prefix=gen_prefix, tight_factor=tight_factor, gpu_no=gpu_no, verbose=verbose)
    mod = UCMPModel{T,TD,TI,TM}(env; start_period=start_period, end_period=end_period, ramp_ratio=ramp_ratio)

    n = mod.models[1].n
    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    env.params.shmem_size = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)
    env.params.gen_shmem_size = sizeof(Float64)*(14*3+3*3^2) + sizeof(Int)*(4*3)

    # # For warm start, solve each time period without ramp constraints.
    # if warm_start
    #     for i=1:mod.len_horizon
    #         admm_restart(env, mod.models[i])
    #     end
    #     init_solution!(mod, mod.solution, rho_pq, rho_va)
    # end

    # admm_restart(env, mod)

    return env, mod
end