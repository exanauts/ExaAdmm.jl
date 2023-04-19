function admm_one_level(
    env::AdmmEnv, mod::AbstractOPFModel
)
    par = env.params
    info = mod.info
    sol = mod.solution

    sqrt_d = sqrt(mod.nvar)
    OUTER_TOL = sqrt_d*(par.outer_eps) #adjusted outer loop tolerance

    fill!(info, 0)
    info.mismatch = Inf

    #eliminate second level
    info.norm_z_prev = info.norm_z_curr = 0
    par.initial_beta = 0
    par.beta = 0
    sol.lz .= 0
    sol.z_curr .= 0
    sol.z_prev .= 0
    par.inner_iterlim = 1

    if par.verbose > 0
        admm_update_residual(env, mod)
        @printf("%8s  %10s  %10s  %10s  %10s %10s %10s\n",
                "Iter", "Objval", "Auglag", "PrimRes", "PrimTol", "DualRes", "DualTol")

        @printf("%8d  %10.3e  %10.3e  %10.3e  %10.3e %10.3e  %10.3e\n",
                info.outer, info.objval, info.auglag, info.mismatch, OUTER_TOL, info.dualres, OUTER_TOL*norm(sol.rho)/sqrt_d)
    end

    info.status = :IterationLimit

    overall_time = @timed begin
    while info.outer < par.outer_iterlim
        admm_increment_outer(env, mod)


        admm_increment_reset_inner(env, mod)
        while info.inner < par.inner_iterlim
            admm_increment_inner(env, mod)

            admm_update_x(env, mod)

            admm_update_xbar(env, mod)

            admm_update_l_single(env, mod)

            admm_update_residual(env, mod)

            if par.verbose > 0
                if (info.cumul % 50) == 0
                    @printf("%8s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Iter", "Objval", "Auglag", "PrimRes", "PrimTol", "DualRes", "DualTol")
                end

                @printf("%8d  %10.3e  %10.3e  %10.3e  %10.3e %10.3e  %10.3e\n",
                        info.outer, info.objval, info.auglag, info.mismatch, OUTER_TOL, info.dualres, OUTER_TOL*norm(sol.rho)/sqrt_d)
            end

        end # while inner

        # mismatch: x-xbar
        if info.mismatch <= OUTER_TOL && info.dualres <= OUTER_TOL*norm(sol.rho)/sqrt_d
            info.status = :Solved
            break
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
