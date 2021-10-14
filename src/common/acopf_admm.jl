function solve_acopf(case::String;
    case_format="matpower",
    outer_iterlim=10, inner_iterlim=800, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4,
    use_gpu=false, use_linelimit=true, tight_factor=0.99,
    outer_eps=2*1e-4, solve_pf=false, gpu_no=0, verbose=1)

    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit, use_twolevel=false,
            tight_factor=tight_factor, solve_pf=solve_pf, gpu_no=gpu_no, verbose=verbose)
    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim

    mod = Model{T,TD,TI,TM}(env)

    if use_gpu
        # Set rateA in membuf.
        CUDA.@sync @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
    else
        mod.membuf[29,:] .= mod.rateA
    end

    admm_restart(env, mod)
    return env, mod
end
