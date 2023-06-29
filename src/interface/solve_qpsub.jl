function solve_qpsub(
    case::String,
    Hs,
    LH_1h,
    RH_1h,
    LH_1i,
    RH_1i,
    LH_1j,
    RH_1j,
    LH_1k,
    RH_1k,
    ls,
    us,
    pgmax,
    pgmin,
    qgmax,
    qgmin,
    c1,
    c2,
    Pd,
    Qd,
    initial_beta;
    case_format = "matpower",
    outer_iterlim = 20,
    inner_iterlim = 1000,
    rho_pq = 400.0,
    rho_va = 40000.0,
    obj_scale = 1.0,
    scale = 1e-4,
    storage_ratio = 0.0,
    storage_charge_max = 1.0,
    use_gpu = false,
    ka_device = nothing,
    use_linelimit = true,
    use_projection = false,
    tight_factor = 1.0,
    outer_eps = 2 * 1e-4,
    gpu_no = 0,
    verbose = 1,
    onelevel = true,
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
    elseif has_cuda_gpu()
            TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    elseif has_rocm_gpu()
        TD = ROCArray{Float64,1}; TI = ROCArray{Int,1}; TM = ROCArray{Float64,2}
    else
        error("Inconsistent device selection use_gpu=$use_gpu and ka_device=$(typeof(ka_device))")
    end

    env = AdmmEnv{T,TD,TI,TM}(
        case,
        rho_pq,
        rho_va;
        case_format = case_format,
        use_gpu = use_gpu,
        ka_device = ka_device,
        use_linelimit = use_linelimit,
        use_projection = use_projection,
        tight_factor = tight_factor,
        gpu_no = gpu_no,
        storage_ratio = storage_ratio,
        storage_charge_max = storage_charge_max,
        verbose = verbose,
    )
    mod = ModelQpsub{T,TD,TI,TM}(env)

    data = mod.grid_data

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
    env.params.shmem_size = sizeof(Float64) * (16 * mod.n + 4 * mod.n^2 + 178) + sizeof(Int) * (4 * mod.n)
    env.params.initial_beta = initial_beta #use my initial beta

    init_solution!(
        mod,
        mod.solution,
        env.initial_rho_pq,
        env.initial_rho_va,
        isa(ka_device, KA.CPU) ? nothing : ka_device
    )

    #one-level or two-level admm
    if onelevel
        admm_one_level(env, mod, isa(ka_device, KA.CPU) ? nothing : ka_device)
    else
        @warn "two-level ADMM is not implemented in QPsub"
    end
    return env, mod
end
