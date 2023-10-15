function admm_one_level(
    env::AdmmEnv, mod::AbstractOPFModel,
    device = nothing
)
    par = env.params
    info = mod.info
    sol = mod.solution

    info.primtol = par.RELTOL
    info.dualtol = par.RELTOL

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
        admm_update_residual(env, mod, device)
        @printf("%8s  %10s  %10s  %10s  %10s  %10s  %10s %10s %10s %10s\n",
                "Iter", "Objval", "Auglag", "PrimScal", "PrimRes", "PrimTol", "DualScal", "DualRes", "DualTol", "rho")

        @printf("%8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e %10.3e  %10.3e  %10.3e\n",
                info.outer, info.objval, info.auglag, info.primsca, info.primres, info.primtol, info.dualsca, info.dualres, info.dualtol, sol.rho[1])
    end

    info.status = :IterationLimit

    overall_time = @timed begin
    while info.outer < par.outer_iterlim
        admm_increment_outer(env, mod, device)


        admm_increment_reset_inner(env, mod)
        while info.inner < par.inner_iterlim
            admm_increment_inner(env, mod, device)

            admm_update_x(env, mod, device)

            admm_update_xbar(env, mod, device)

            admm_update_l_single(env, mod, device)

            admm_update_residual(env, mod, device)

            if par.verbose > 0
                if (info.cumul % 50) == 0
                    @printf("%8s  %10s  %10s  %10s  %10s  %10s  %10s %10s %10s %10s\n",
                            "Iter", "Objval", "Auglag", "PrimScal", "PrimRes", "PrimTol", "DualScal", "DualRes", "DualTol", "rho")
                end

                @printf("%8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e %10.3e  %10.3e  %10.3e\n",
                info.outer, info.objval, info.auglag, info.primsca, info.primres, info.primtol, info.dualsca, info.dualres, info.dualtol, sol.rho[1])
            end

        end # while inner

        # mismatch: x-xbar
        if info.primres <= info.primtol # && info.dualres <= info.dualtol
            info.status = :Solved
            break
        end

        # residual balancing
        if par.rb_switch
            if info.primres > par.rb_beta1 * info.dualres
                sol.rho .= min.(sol.rho * par.rb_tau, 1e+16)
            elseif par.rb_beta2 * info.primres < info.dualres
                sol.rho .= max.(sol.rho / par.rb_tau, 1e-16)
            end
        end

    end # while outer
    end # @timed

    info.time_overall = overall_time.time
    admm_poststep(env, mod, device)

    if par.verbose > 0
        print_statistics(env, mod)
    end

    return
end
