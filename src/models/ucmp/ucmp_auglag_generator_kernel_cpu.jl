#########################################################
# Table of variables and parameters
        # l: lambda; r_l: lambda for ramp variables; 
        # uc_l: lambda for uc variables;
        #
        # x[1]      : p_{t,I}
        # x[2]      : q_{t,I}
        # x[3]      : phat_{t,I}
        # x[4]      : v_{t,I}
        # x[5]      : w_{t,I}
        # x[6]      : y_{t,I}
        # x[7]      : vhat_{t,I}
        # x[8]      : s^U_{t,I}
        # x[9]      : s^D_{t,I}
        # x[10]     : s^{p,UB}_{t,I}
        # x[11]     : s^{p,LB}_{t,I}
        # x[12]     : s^{q,UB}_{t,I}
        # x[13]     : s^{q,LB}_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{q_{t,I}}
        # param[3,I]: lambda_{phat_{t,I}}
        # param[4,I]: lambda_{v_{t,I}}
        # param[5,I]: lambda_{w_{t,I}}
        # param[6,I]: lambda_{y_{t,I}}
        # param[7,I]: lambda_{vhat_{t,I}}
        # param[8,I]: lambda_{su_{t,I}}
        # param[9,I]: lambda_{sd_{t,I}}
        # param[10,I]: lambda_{pUB_{t,I}}
        # param[11,I]: lambda_{pLB_{t,I}}
        # param[12,I]: lambda_{qUB_{t,I}}
        # param[13,I]: lambda_{qLB_{t,I}}
        # param[14,I]: rho_{p_{t,I}}
        # param[15,I]: rho_{q_{t,I}}
        # param[16,I]: rho_{phat_{t,I}}
        # param[17,I]: rho_{v_{t,I}}
        # param[18,I]: rho_{w_{t,I}}
        # param[19,I]: rho_{y_{t,I}}
        # param[20,I]: rho_{vhat_{t,I}}
        # param[21,I]: rho_{su_{t,I}}
        # param[22,I]: rho_{sd_{t,I}}
        # param[23,I]: rho_{pUB_{t,I}}
        # param[24,I]: rho_{pLB_{t,I}}
        # param[25,I]: rho_{qUB_{t,I}}
        # param[26,I]: rho_{qLB_{t,I}}
        # param[27,I]: p_{t,I(i)} - z_{p_{t,I}}
        # param[28,I]: q_{t,I(i)} - z_{q_{t,I}}
        # param[29,I]: p_{t-1,I(i)} - z_{phat_{t-1,I}}
        # param[30,I]: vbar_{t,I} - z_{v_{t,I}}
        # param[31,I]: wbar_{t,I} - z_{w_{t,I}}
        # param[32,I]: ybar_{t,I} - z_{y_{t,I}}
        # param[33,I]: vbar_{t-1,I} - z_{vhat_{t,I}}
        # param[34,I]: RUg
        # param[35,I]: SUg
        # param[36,I]: RDg
        # param[37,I]: SDg
        # param[38,I]: pmax
        # param[39,I]: pmin
        # param[40,I]: qmax
        # param[41,I]: qmin
#########################################################

function ucmp_auglag_generator_kernel(
    t::Int, n::Int, ngen::Int, gen_start::Int,
    major_iter::Int, max_auglag::Int, xi_max::Float64, scale::Float64,
    u::Array{Float64,1}, v::Array{Float64,1}, z::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1},
    r_u::Array{Float64,1}, r_v::Array{Float64,1}, r_z::Array{Float64,1},
    r_l::Array{Float64,1}, r_rho::Array{Float64,1}, r_s::Array{Float64,1},
    vr_u::Array{Float64,1}, vr_v::Array{Float64,1}, vr_z::Array{Float64,1},
    vr_l::Array{Float64,1}, vr_rho::Array{Float64,1}, vr_s::Array{Float64,1},
    uc_u::Array{Float64,2}, uc_v::Array{Float64,2}, uc_z::Array{Float64,2},
    uc_l::Array{Float64,2}, uc_rho::Array{Float64,2}, uc_s::Array{Float64,2},
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
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        v_idx = 3*(t-1) + 1
        w_idx = 3*(t-1) + 2
        y_idx = 3*t

        c2 = _c2[I]; c1 = _c1[I]; c0 = _c0[I]

        xl[1] = xl[3] = pgmin[I]
        xl[2] = qgmin[I]
        xl[8] = xl[9] = -4*ramp_limit[I]
        xl[10] = xl[11] = -abs(pgmax[I])-abs(pgmin[I])
        xl[12] = xl[13] = -abs(qgmax[I])-abs(qgmin[I])
        # COMMENTED: THESE BOUNDS ARE TOO TIGHT FOR S VARIABLES. LEAD TO DIVERGENCE FOR 9-BUS CASE.
        # xl[8] = xl[9] = -2*ramp_limit[I]
        # xl[10] = xl[11] = -pgmax[I]
        # xl[12] = xl[13] = -qgmax[I]
        xu[1] = xu[3] = pgmax[I]
        xu[2] = qgmax[I]
        xu[4] = xu[5] = xu[6] = xu[7] = 1
        xu[8] = xu[9] = xu[10] = xu[11] = xu[12] = xu[13] = 0

        x[1] = min(xu[1], max(xl[1], u[pg_idx]))
        x[2] = min(xu[2], max(xl[2], u[qg_idx]))
        x[3] = min(xu[3], max(xl[3], r_u[I]))
        x[4] = uc_u[I, v_idx]
        x[5] = uc_u[I, w_idx]
        x[6] = uc_u[I, y_idx]
        t > 1 ? x[7] = vr_u[I] : x[7] = 0.
        x[8] = min(xu[8], max(xl[8], r_s[2*I-1]))
        x[9] = min(xu[9], max(xl[9], r_s[2*I]))
        x[10] = min(xu[10], max(xl[10], uc_s[I, 4*t-3]))
        x[11] = min(xu[11], max(xl[11], uc_s[I, 4*t-2]))
        x[12] = min(xu[12], max(xl[12], uc_s[I, 4*t-1]))
        x[13] = min(xu[13], max(xl[13], uc_s[I, 4*t]))

        param[1,I] = l[pg_idx]
        param[2,I] = l[qg_idx]
        param[3,I] = r_l[I]
        param[4,I] = uc_l[I, v_idx]
        param[5,I] = uc_l[I, w_idx]
        param[6,I] = uc_l[I, y_idx]
        param[7,I] = vr_l[I]
        # param[8,I] = r_l[4*I-1]
        # param[9,I] = r_l[4*I]
        # param[10,I] = uc_l[I, 7*t-3]
        # param[11,I] = uc_l[I, 7*t-2]
        # param[12,I] = uc_l[I, 7*t-1]
        # param[13,I] = uc_l[I, 7*t]
        param[14,I] = rho[pg_idx]
        param[15,I] = rho[qg_idx]
        param[16,I] = r_rho[I]
        param[17,I] = uc_rho[I, v_idx]
        param[18,I] = uc_rho[I, w_idx]
        param[19,I] = uc_rho[I, y_idx]
        param[20,I] = vr_rho[I]
        # param[21,I] = r_rho[4*I-1]
        # param[22,I] = r_rho[4*I]
        # param[23,I] = uc_rho[7*I-3]
        # param[24,I] = uc_rho[7*I-2]
        # param[25,I] = uc_rho[7*I-1]
        # param[26,I] = uc_rho[7*I]
        param[27,I] = v[pg_idx] - z[pg_idx]
        param[28,I] = v[qg_idx] - z[qg_idx]
        t > 1 ? param[29,I] = r_v[I] - r_z[I] : param[29,I] = 0.
        param[30,I] = uc_v[I, v_idx] - uc_z[I, v_idx]
        param[31,I] = uc_v[I, w_idx] - uc_z[I, w_idx]
        param[32,I] = uc_v[I, y_idx] - uc_z[I, y_idx]
        t > 1 ? param[33,I] = uc_v[I, v_idx-3] - vr_z[I] : param[33,I] = 0.
        param[34,I] = ramp_limit[I]
        param[35,I] = ramp_limit[I]
        param[36,I] = ramp_limit[I]
        param[37,I] = ramp_limit[I]
        param[38,I] = pgmax[I]
        param[39,I] = pgmin[I]
        param[40,I] = qgmax[I]
        param[41,I] = qgmin[I]

        # Initialization of Augmented Lagrangian Method Parameters
        if major_iter <= 1
            # mu (Lagrangian term parameter)
            param[10,I] = 10.0
            param[11,I] = 10.0
            param[12,I] = 10.0
            param[13,I] = 10.0
            # xi (augmented term parameter)
            param[23,I] = 10.0
            param[24,I] = 10.0
            param[25,I] = 10.0
            param[26,I] = 10.0
            if t > 1
                # mu
                param[8,I] = 10.0
                param[9,I] = 10.0
                # xi
                param[21,I] = 10.0
                param[22,I] = 10.0    
            end
        # else
        #     xi = param[43,I]
        end

        function eval_f_cb(x)
            f = eval_f_generator_continuous_kernel_cpu(t, I, x, param, scale, c2, c1, c0, baseMVA)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_generator_continuous_kernel_cpu(t, I, x, g, param, scale, c2, c1, c0, baseMVA)
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_generator_continuous_kernel_cpu(t, I, x, mode, scale, rows, cols, lambda, values,
                param, c2, c1, c0, baseMVA)
            return
        end

        t > 1 ? _st = 1 : _st = 3
        etas = zeros(6)
        omegas = zeros(6)
        for i in _st:6
            etas[i] = 1 / param[20+i,I]^0.1
            omegas[i] = 1 / param[20+i,I]
        end
        gtol = 1e-6

        nele_hess = 41
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
            pUB_cviol = x[1] - pgmax[I]*x[4] - x[10]
            pLB_cviol = x[1] - pgmin[I]*x[4] + x[11]
            qUB_cviol = x[2] - qgmax[I]*x[4] - x[12]
            qLB_cviol = x[2] - qgmin[I]*x[4] + x[13]

            cvios = [0., 0., pUB_cviol, pLB_cviol, qUB_cviol, qLB_cviol]
            if t > 1
                cvios[1] = x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8]
                cvios[2] = x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9]
            end

            terminate = true
            for i in _st:6
                if abs(cvios[i]) <= etas[i]
                    if abs(cvios[i]) > 1e-6
                        terminate = false
                        param[7+i,I] += param[20+i,I]*cvios[i]                        
                        etas[i] = etas[i] / param[20+i,I]^0.9
                        omegas[i] = omegas[i] / param[20+i,I]
                    end
                else
                    terminate = false
                    param[20+i,I] = min(xi_max, param[20+i,I]*10)
                    etas[i] = 1 / param[20+i,I]^0.1
                    omegas[i] = 1 / param[20+i,I]
                end
            end

            if it >= max_auglag
                # println("inner auglag reach max iteration at time $(t) for generator $(I)")
                terminate = true
            end
        end

        u[pg_idx] = x[1]
        u[qg_idx] = x[2]
        uc_u[I, v_idx] = x[4]
        uc_u[I, w_idx] = x[5]
        uc_u[I, y_idx] = x[6]
        uc_s[I, 4*t-3] = x[10]
        uc_s[I, 4*t-2] = x[11]
        uc_s[I, 4*t-1] = x[12]
        uc_s[I, 4*t] = x[13]

        if t > 1
            r_u[I] = x[3]
            vr_u[I] = x[7]
            r_s[2*I-1] = x[8]
            r_s[2*I] = x[9]
        end

    end
end