function solve_ucmp(case::String, load_prefix::String, gen_prefix::String;
    case_format="matpower", start_period=1, end_period=1,
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, ka_device=nothing, use_linelimit=true, use_projection=false, tight_factor=1.0,
    outer_eps=2*1e-4, gpu_no=0, verbose=1, ramp_ratio=0.02, warm_start=true, multiperiod_tight=true
)

    T = Float64
    if !use_gpu && (isa(ka_device, Nothing) || isa(ka_device, KA.CPU))
        TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
        ka_device = nothing
    elseif use_gpu && isa(ka_device, Nothing)
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    elseif use_gpu && isa(ka_device, KA.Device)
        if has_cuda_gpu()
            TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
        elseif has_rocm_gpu()
            TD = ROCArray{Float64,1}; TI = ROCArray{Int,1}; TM = ROCArray{Float64,2}
        end
    else
        error("Inconsistent device selection use_gpu=$use_gpu and ka_device=$(typepof(ka_device))")
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; load_prefix=load_prefix, case_format=case_format,
            use_gpu=use_gpu, ka_device=ka_device, use_linelimit=use_linelimit,
            use_projection=use_projection, tight_factor=tight_factor, gpu_no=gpu_no,
            storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
            verbose=verbose)
    mod = UCMPModel{T,TD,TI,TM}(env, gen_prefix; start_period=start_period, end_period=end_period, ramp_ratio=ramp_ratio)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim

    # For warm start, solve each time period without ramp constraints.
    if warm_start
        for i=1:mod.mpmodel.len_horizon
            admm_two_level(env, mod.mpmodel.models[i])
        end
        init_solution!(mod, mod.uc_solution, rho_pq, rho_va)
    end

    admm_two_level(env, mod)

    return env, mod
end