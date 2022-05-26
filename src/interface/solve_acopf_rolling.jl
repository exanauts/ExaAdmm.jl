function solve_acopf_rolling(case::String, load_prefix::String;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4,
    use_gpu=false, use_linelimit=true, use_projection=false,
    tight_factor=0.99, outer_eps=2*1e-4, gpu_no=0, verbose=1,
    ramp_ratio=0.02, start_period=1, end_period=6, result_file="warm-start")

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
            use_projection=use_projection, load_prefix=load_prefix,
            tight_factor=tight_factor, gpu_no=gpu_no, verbose=verbose)
    mod = Model{T,TD,TI,TM}(env; ramp_ratio=ramp_ratio)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim

    admm_restart_rolling(env, mod; start_period=start_period, end_period=end_period, result_file=result_file)
    return env, mod
end
