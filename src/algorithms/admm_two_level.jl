function admm_two_level(
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
        admm_update_residual(env, mod)
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
        admm_increment_outer(env, mod)
        admm_outer_prestep(env, mod)

        admm_increment_reset_inner(env, mod)
        while info.inner < par.inner_iterlim
            admm_increment_inner(env, mod)
            admm_inner_prestep(env, mod)

            admm_update_x(env, mod)
            admm_update_xbar(env, mod)
            admm_update_z(env, mod)
            admm_update_l(env, mod)
            admm_update_residual(env, mod)

            # an adjusting termination criteria for inner loop (i.e., inner loop is not solved to exact)
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
            
            # primres: x-xbar+z_curr
            if info.primres <= info.eps_pri #|| info.dualres <= par.DUAL_TOL
                break
            end
        end # while inner
        
        # mismatch: x-xbar
        if info.mismatch <= OUTER_TOL
            info.status = :Solved
            break
        end

        admm_update_lz(env, mod)
        
        # if z_curr too large vs z_prev, increase penalty
        if info.norm_z_curr > par.theta*info.norm_z_prev
            par.beta = min(par.inc_c*par.beta, 1e24)
        end
    end # while outer
    end # @timed

    info.time_overall = overall_time.time
    admm_poststep(env, mod)

    if par.verbose > 0
        print_statistics(env, mod)
    end

    return
end