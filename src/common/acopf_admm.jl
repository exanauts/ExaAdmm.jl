function solve_acopf(case::String;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, use_linelimit=true, use_projection=false, tight_factor=0.99,
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
    mod = Model{T,TD,TI,TM}(env)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim
    env.params.shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    admm_restart(env, mod)

    return env, mod
end

function admm_restart(
    env::AdmmEnv, mod::AbstractOPFModel
)
    par = env.params
    info = mod.info

    sqrt_d = sqrt(mod.nvar)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    fill!(info, 0)
    info.mismatch = Inf
    info.norm_z_prev = info.norm_z_curr = Inf
    par.beta = par.initial_beta

    if par.verbose > 0
        acopf_admm_update_residual(env, mod)
        @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                "Outer", "Inner", "Objval", "AugLag", "PrimRes", "EpsPrimRes",
                "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")

        @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                info.outer, info.inner, info.objval, info.auglag, info.primres, info.eps_pri,
                info.dualres, info.norm_z_curr, info.mismatch, OUTER_TOL, par.beta)
    end

    info.status = :IterationLimit

    overall_time = @timed begin
    while info.outer < par.outer_iterlim
        acopf_admm_increment_outer(env, mod)
        acopf_admm_outer_prestep(env, mod)

        acopf_admm_increment_reset_inner(env, mod)
        while info.inner < par.inner_iterlim
            acopf_admm_increment_inner(env, mod)
            acopf_admm_inner_prestep(env, mod)

            acopf_admm_update_x(env, mod)
            acopf_admm_update_xbar(env, mod)
            acopf_admm_update_z(env, mod)
            acopf_admm_update_l(env, mod)
            acopf_admm_update_residual(env, mod)

            info.eps_pri = sqrt_d / (2500*info.outer)

            if par.verbose > 0
                if (info.cumul % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "Objval", "AugLag", "PrimRes", "EpsPrimRes",
                            "DualRes", "||z||", "||Ax+By||", "OuterTol", "Beta")
                end

                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        info.outer, info.inner, info.objval, info.auglag, info.primres, info.eps_pri,
                        info.dualres, info.norm_z_curr, info.mismatch, OUTER_TOL, par.beta)
            end

            if info.primres <= info.eps_pri #|| info.dualres <= par.DUAL_TOL
                break
            end
        end # while inner

        if info.mismatch <= OUTER_TOL
            info.status = :Solved
            break
        end

        acopf_admm_update_lz(env, mod)

        if info.norm_z_curr > par.theta*info.norm_z_prev
            par.beta = min(par.inc_c*par.beta, 1e24)
        end
    end # while outer
    end # @timed

    info.time_overall = overall_time.time
    acopf_admm_poststep(env, mod)

    if par.verbose > 0
        print_statistics(env, mod)
    end

    return
end