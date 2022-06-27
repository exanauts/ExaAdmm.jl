function solve_acopf(case::String;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, ka_device=nothing, use_linelimit=true, use_projection=false, tight_factor=1.0,
    outer_eps=2*1e-4, gpu_no=0, verbose=1
)
    T = Float64
    # 1. ka_device = nothing and use_gpu = false, CPU version of the code is used
    # 2. ka_device = KA.CPU() and use_gpu = false, CPU version of the code is used, NOT the KA.CPU kernels
    #    due to nested kernels limitations and no added benefit
    # 3. ka_device = nothing and use_gpu = true, use original CUDA.jl kernels
    # 4. ka_device is a KA.GPU and use_gpu = true, use KA kernels
    if !use_gpu && (isa(ka_device, Nothing) || isa(ka_device, KA.CPU))
        TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
        ka_device = nothing
    elseif use_gpu && isa(ka_device, Nothing)
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    elseif use_gpu && isa(ka_device, KA.Device)
        if CUDA.has_cuda_gpu()
            TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
        elseif AMDGPU.has_rocm_gpu()
            TD = ROCArray{Float64,1}; TI = ROCArray{Int,1}; TM = ROCArray{Float64,2}
        end
    else
        error("Inconsistent device selection use_gpu=$use_gpu and ka_device=$(typepof(ka_device))")
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, ka_device=ka_device, use_linelimit=use_linelimit,
            use_projection=use_projection, tight_factor=tight_factor, gpu_no=gpu_no,
            storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
            verbose=verbose)
    mod = ModelAcopf{T,TD,TI,TM}(env)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim

    admm_two_level(env, mod, isa(ka_device, KA.CPU) ? nothing : ka_device)

    return env, mod
end
