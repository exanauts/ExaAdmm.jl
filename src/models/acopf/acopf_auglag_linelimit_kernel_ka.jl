@kernel function auglag_linelimit_two_level_alternative_ka(
        N::Val{n}, nline::Int, line_start::Int,
        major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
        u, xbar,
        z, l, rho,
        shift_lines::Int, param,
        _YffR, _YffI,
        _YftR, _YftI,
        _YttR, _YttI,
        _YtfR, _YtfI,
        frVmBound, toVmBound,
        frVaBound, toVaBound
    ) where {n}

    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J
    id_line = I + shift_lines

    x = @localmem Float64 (n,)
    xl = @localmem Float64 (n,)
    xu = @localmem Float64 (n,)

    @inbounds begin
        YffR = _YffR[id_line]; YffI = _YffI[id_line]
        YftR = _YftR[id_line]; YftI = _YftI[id_line]
        YttR = _YttR[id_line]; YttI = _YttI[id_line]
        YtfR = _YtfR[id_line]; YtfI = _YtfI[id_line]

        pij_idx = line_start + 8*(id_line-1)

        xl[1] = frVmBound[2*id_line-1]
        xu[1] = frVmBound[2*id_line]
        xl[2] = toVmBound[2*id_line-1]
        xu[2] = toVmBound[2*id_line]
        xl[3] = frVaBound[2*id_line-1]
        xu[3] = frVaBound[2*id_line]
        xl[4] = toVaBound[2*id_line-1]
        xu[4] = toVaBound[2*id_line]
        xl[5] = -param[29,id_line]
        xu[5] = 0.0
        xl[6] = -param[29,id_line]
        xu[6] = 0.0

        x[1] = min(xu[1], max(xl[1], sqrt(u[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], sqrt(u[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u[pij_idx+7]))
        x[5] = min(xu[5], max(xl[5], -(u[pij_idx]^2 + u[pij_idx+1]^2)))
        x[6] = min(xu[6], max(xl[6], -(u[pij_idx+2]^2 + u[pij_idx+3]^2)))

        param[1,id_line] = l[pij_idx]
        param[2,id_line] = l[pij_idx+1]
        param[3,id_line] = l[pij_idx+2]
        param[4,id_line] = l[pij_idx+3]
        param[5,id_line] = l[pij_idx+4]
        param[6,id_line] = l[pij_idx+5]
        param[7,id_line] = l[pij_idx+6]
        param[8,id_line] = l[pij_idx+7]
        param[9,id_line] = rho[pij_idx]
        param[10,id_line] = rho[pij_idx+1]
        param[11,id_line] = rho[pij_idx+2]
        param[12,id_line] = rho[pij_idx+3]
        param[13,id_line] = rho[pij_idx+4]
        param[14,id_line] = rho[pij_idx+5]
        param[15,id_line] = rho[pij_idx+6]
        param[16,id_line] = rho[pij_idx+7]
        param[17,id_line] = xbar[pij_idx] - z[pij_idx]
        param[18,id_line] = xbar[pij_idx+1] - z[pij_idx+1]
        param[19,id_line] = xbar[pij_idx+2] - z[pij_idx+2]
        param[20,id_line] = xbar[pij_idx+3] - z[pij_idx+3]
        param[21,id_line] = xbar[pij_idx+4] - z[pij_idx+4]
        param[22,id_line] = xbar[pij_idx+5] - z[pij_idx+5]
        param[23,id_line] = xbar[pij_idx+6] - z[pij_idx+6]
        param[24,id_line] = xbar[pij_idx+7] - z[pij_idx+7]

        if major_iter == 1
            param[27,id_line] = 10.0
            mu = 10.0
        else
            mu = param[27,id_line]
        end

        @synchronize

        eta = 1.0 / mu^0.1
        omega = 1.0 / mu

        it = 0
        terminate = false

        while !terminate
            it += 1

            # Solve the branch problem.
            status, minor_iter = tron_linelimit_kernel(n, shift_lines, 500, 200, 1e-6, scale, true, x, xl, xu,
                                                       param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)

            # Check the termination condition.
            vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
            vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
            pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

            cviol1 = pij^2 + qij^2 + x[5]
            cviol2 = pji^2 + qji^2 + x[6]

            cnorm = max(abs(cviol1), abs(cviol2))

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    if tx == 1
                        param[25,id_line] += mu*cviol1
                        param[26,id_line] += mu*cviol2
                    end
                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[27,id_line] = mu
            end

            if it >= max_auglag
                terminate = true
            end

            @synchronize
        end

        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        u[pij_idx] = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        u[pij_idx+1] = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        u[pij_idx+2] = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        u[pij_idx+3] = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
        u[pij_idx+4] = x[1]^2
        u[pij_idx+5] = x[2]^2
        u[pij_idx+6] = x[3]
        u[pij_idx+7] = x[4]
        param[27,id_line] = mu

        @synchronize
    end
end
