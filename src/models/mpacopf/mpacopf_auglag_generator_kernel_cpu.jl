function auglag_generator_kernel(
    n::Int, ngen::Int, gen_start::Int,
    major_iter::Int, max_auglag::Int, xi_max::Float64, scale::Float64,
    u::Array{Float64,1}, v::Array{Float64,1}, z::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1},
    r_u::Array{Float64,1}, r_v::Array{Float64,1}, r_z::Array{Float64,1},
    r_l::Array{Float64,1}, r_rho::Array{Float64,1}, r_s::Array{Float64,1},
    param::Array{Float64,2},
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    ramp_limit::Array{Float64,1},
    _c2::Array{Float64,1}, _c1::Array{Float64,1}, _c0::Array{Float64,1}, baseMVA::Float64
)
    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)

    @inbounds for I=1:ngen
        # x[1]      : p_{t,I}
        # x[2]      : phat_{t-1,I}
        # x[3]      : s_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{phat_{t-1,I}}
        # param[3,I]: rho_{p_{t,I}}
        # param[4,I]: rho_{phat_{t-1,I}}
        # param[5,I]: pbar_{t,I} - z_{p_{t,I}}
        # param[6,I]: pbar_{t-1,I} - z_{phat_{t-1,I}}
        # param[7,I]: mu
        # param[8,I]: xi

        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx]+rho[qg_idx]*(-v[qg_idx]+z[qg_idx]))) / rho[qg_idx]))

        c2 = _c2[I]; c1 = _c1[I]; c0 = _c0[I]
        xl[1] = xl[2] = pgmin[I]
        xu[1] = xu[2] = pgmax[I]
        xl[3] = -ramp_limit[I]
        xu[3] = ramp_limit[I]

        x[1] = min(xu[1], max(xl[1], u[pg_idx]))
        x[2] = min(xu[2], max(xl[2], r_u[I]))
        x[3] = min(xu[3], max(xl[3], r_s[I]))

        param[1,I] = l[pg_idx]
        param[2,I] = r_l[I]
        param[3,I] = rho[pg_idx]
        param[4,I] = r_rho[I]
        param[5,I] = v[pg_idx] - z[pg_idx]
        param[6,I] = r_v[pg_idx] - r_z[I]

        if major_iter <= 1
            param[8,I] = 10.0
            xi = 10.0
        else
            xi = param[8,I]
        end

        function eval_f_cb(x)
            f = eval_f_generator_kernel_cpu(I, x, param, scale, c2, c1, c0, baseMVA)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_generator_kernel_cpu(I, x, g, param, scale, c2, c1, c0, baseMVA)
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_generator_kernel_cpu(I, x, mode, scale, rows, cols, lambda, values,
                param, c2, c1, c0, baseMVA)
            return
        end

        eta = 1 / xi^0.1
        omega = 1 / xi
        gtol = 1e-6

        nele_hess = 6
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)
        tron.x .= x
        it = 0
        avg_tron_minor = 0
        terminate = false

        while !terminate
            it += 1

            ExaTron.solveProblem(tron)
            x .= tron.x
            avg_tron_minor += tron.minor_iter

            # Check the termination condition.
            cviol = x[1] - x[2] - x[3]
            cnorm = abs(cviol)

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[7,I] += xi*cviol
                    eta = eta / xi^0.9
                    omega = omega / xi
                end
            else
                xi = min(xi_max, xi*10)
                eta = 1 / xi^0.1
                omega = 1 / xi
                param[8,I] = xi
            end

            if it >= max_auglag
                terminate = true
            end
        end

        u[pg_idx] = x[1]
        r_u[I] = x[2]
        r_s[I] = x[3]
    end
end

function auglag_generator_loose_kernel(
    n::Int, ngen::Int, gen_start::Int,
    major_iter::Int, max_auglag::Int, xi_max::Float64, scale::Float64,
    u::Array{Float64,1}, v::Array{Float64,1}, z::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1},
    r_u::Array{Float64,1}, r_v::Array{Float64,1}, r_z::Array{Float64,1},
    r_l::Array{Float64,1}, r_rho::Array{Float64,1}, r_s::Array{Float64,1},
    param::Array{Float64,2},
    pgmin::Array{Float64,1}, pgmax::Array{Float64,1},
    qgmin::Array{Float64,1}, qgmax::Array{Float64,1},
    ramp_limit::Array{Float64,1},
    _c2::Array{Float64,1}, _c1::Array{Float64,1}, _c0::Array{Float64,1}, baseMVA::Float64
)
    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)

    @inbounds for I=1:ngen
        # x[1]      : p_{t,I}
        # x[2]      : phat_{t-1,I}
        # x[3]      : s_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{ptilde_{t,I}}
        # param[3,I]: lambda_{phat_{t-1,I}}
        # param[4,I]: rho_{p_{t,I}}
        # param[5,I]: rho_{ptilde_{t,I}}
        # param[6,I]: rho_{phat_{t-1,I}}
        # param[7,I]: pbar_{t,I} - z_{p_{t,I}}
        # param[8,I]: ptilde_{t,I} - z_{ptilde_{t,I}}
        # param[9,I]: ptilde_{t-1,I} - z_{phat_{t-1,I}}
        # param[10,I]: mu
        # param[11,I]: xi

        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1

        u[qg_idx] = max(qgmin[I],
                        min(qgmax[I],
                            (-(l[qg_idx]+rho[qg_idx]*(-v[qg_idx]+z[qg_idx]))) / rho[qg_idx]))

        c2 = _c2[I]; c1 = _c1[I]; c0 = _c0[I]
        xl[1] = xl[2] = pgmin[I]
        xu[1] = xu[2] = pgmax[I]
        xl[3] = -ramp_limit[I]
        xu[3] = ramp_limit[I]

        x[1] = min(xu[1], max(xl[1], u[pg_idx]))
        x[2] = min(xu[2], max(xl[2], r_u[I]))
        x[3] = min(xu[3], max(xl[3], r_s[I]))

        param[1,I] = l[pg_idx]
        param[2,I] = r_l[I]
        param[3,I] = r_l[ngen+I]
        param[4,I] = rho[pg_idx]
        param[5,I] = r_rho[I]
        param[6,I] = r_rho[ngen+I]
        param[7,I] = v[pg_idx] - z[pg_idx]
        param[8,I] = r_v[I] - r_z[I]
        param[9,I] = r_v[ngen+I] - r_z[ngen+I]

        if major_iter <= 1
            param[11,I] = 10.0
            xi = 10.0
        else
            xi = param[11,I]
        end

        function eval_f_cb(x)
            f = eval_f_generator_loose_kernel_cpu(I, x, param, scale, c2, c1, c0, baseMVA)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_generator_loose_kernel_cpu(I, x, g, param, scale, c2, c1, c0, baseMVA)
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_generator_kernel_loose_cpu(I, x, mode, scale, rows, cols, lambda, values,
                param, c2, c1, c0, baseMVA)
            return
        end

        eta = 1 / xi^0.1
        omega = 1 / xi
        gtol = 1e-6

        nele_hess = 6
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)
        tron.x .= x
        it = 0
        avg_tron_minor = 0
        terminate = false

        while !terminate
            it += 1

            ExaTron.solveProblem(tron)
            x .= tron.x
            avg_tron_minor += tron.minor_iter

            # Check the termination condition.
            cviol = x[1] - x[2] - x[3]
            cnorm = abs(cviol)

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    param[10,I] += xi*cviol
                    eta = eta / xi^0.9
                    omega = omega / xi
                end
            else
                xi = min(xi_max, xi*10)
                eta = 1 / xi^0.1
                omega = 1 / xi
                param[11,I] = xi
            end

            if it >= max_auglag
                terminate = true
            end
        end

        u[pg_idx] = x[1]
        r_u[I] = x[2]
        r_s[I] = x[3]
    end
end