function auglag_generator_kernel(
    n::Int, ngen::Int, gen_start::Int,
    major_iter::Int, max_auglag::Int, xi_max::Float64, scale::Float64,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    r_u::CuDeviceArray{Float64,1}, r_v::CuDeviceArray{Float64,1}, r_z::CuDeviceArray{Float64,1},
    r_l::CuDeviceArray{Float64,1}, r_rho::CuDeviceArray{Float64,1}, r_s::CuDeviceArray{Float64,1},
    param::CuDeviceArray{Float64,2},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    ramp_limit::CuDeviceArray{Float64,1},
    _c2::CuDeviceArray{Float64,1}, _c1::CuDeviceArray{Float64,1}, _c0::CuDeviceArray{Float64,1}, baseMVA::Float64
)
    tx = threadIdx().x
    I = blockIdx().x

    x = CuDynamicSharedArray(Float64, n)
    xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
    xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))

    @inbounds begin
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

        CUDA.sync_threads()

        eta = 1.0 / xi^0.1
        omega = 1.0 / xi

        it = 0
        terminate = false

        while !terminate
            it += 1

            # Solve the generator problem.
            status, minor_iter = tron_generator_kernel(n, 500, 200, 1e-6, scale, x, xl, xu, param, c2, c1, c0, baseMVA)

            # Check the termination condition.
            cviol = x[1] - x[2] - x[3]
            cnorm = abs(cviol)

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    if tx == 1
                        param[7,I] += xi*cviol
                    end
                    eta = eta / xi^0.9
                    omega  = omega / xi
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

            CUDA.sync_threads()
        end

        u[pg_idx] = x[1]
        r_u[I] = x[2]
        r_s[I] = x[3]

        CUDA.sync_threads()
    end

    return
end
