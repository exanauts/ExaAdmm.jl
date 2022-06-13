"""
    auglag_linelimit_two_level_alternative_qpsub()

- for all line (i,j), update sol.u[pij_idx]
- use auglag_linelimit_two_level_alternative_qpsub_ij()
- TODO: implement for all lines and reuse Youngdae's format (e.g. reuse mu)
"""


"""
   Internal Solution Structure for branch

- branch structure from u (8*nline):   
    - |p_ij   | q_ij  | p_ji   | q_ji    | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji) |
- branch structure for Exatron (8*nline):  
    - | t_ij(linelimit) | t_ji(linelimit) | w_ijR  |  w_ijI   | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)
- Hessian inherited from SQP (6*nline):   
    - |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|   
"""



"""

    Youngdae Auglag Argument vs My Auglag

- n::Int, nline::Int, line_start::Int,
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,

- u::Array{Float64,1}, xbar::Array{Float64,1}, z::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1}, shift_lines::Int,
    param::Array{Float64,2},

- _YffR::Array{Float64,1}, _YffI::Array{Float64,1},
    _YftR::Array{Float64,1}, _YftI::Array{Float64,1},
    _YttR::Array{Float64,1}, _YttI::Array{Float64,1},
    _YtfR::Array{Float64,1}, _YtfI::Array{Float64,1},

- frVmBound::Array{Float64,1}, toVmBound::Array{Float64,1},
    frVaBound::Array{Float64,1}, toVaBound::Array{Float64,1}

- TODO: check frVaBound and toVaBound for slack are 0 in integration

- #! change list:
    - remove n (env.linelimit = true always)
    - shift_lines (=0 unused)
    - (old) param = (new) membuf
    - avoid OPF data format, using SQP data format
"""

function auglag_Ab_linelimit_two_level_alternative_qpsub(
    nline::Int, line_start::Int,
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    u::Array{Float64,1}, xbar::Array{Float64,1}, z::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1}, shift_lines::Int,
    membuf::Array{Float64,2},
    _YffR::Array{Float64,1}, _YffI::Array{Float64,1},
    _YftR::Array{Float64,1}, _YftI::Array{Float64,1},
    _YttR::Array{Float64,1}, _YttI::Array{Float64,1},
    _YtfR::Array{Float64,1}, _YtfI::Array{Float64,1},
    frVmBound::Array{Float64,1}, toVmBound::Array{Float64,1},
    frVaBound::Array{Float64,1}, toVaBound::Array{Float64,1})

    #major_iter: info.inner 

    avg_auglag_it = 0
    avg_minor_it = 0

    x = zeros(n)
    xl = zeros(n)
    xu = zeros(n)

    @inbounds for I=shift_lines+1:shift_lines+nline #shift_lines = 0 (unuse)
        YffR = _YffR[I]; YffI = _YffI[I]
        YftR = _YftR[I]; YftI = _YftI[I]
        YttR = _YttR[I]; YttI = _YttI[I]
        YtfR = _YtfR[I]; YtfI = _YtfI[I]

        pij_idx = line_start + 8*(I-1)

        xl[1] = frVmBound[2*I-1]
        xu[1] = frVmBound[2*I]
        xl[2] = toVmBound[2*I-1]
        xu[2] = toVmBound[2*I]
        xl[3] = frVaBound[2*I-1]
        xu[3] = frVaBound[2*I]
        xl[4] = toVaBound[2*I-1]
        xu[4] = toVaBound[2*I]
        xl[5] = -param[29,I] #slack for line limit
        xu[5] = 0.0
        xl[6] = -param[29,I] #slack for line limit 
        xu[6] = 0.0

        x[1] = min(xu[1], max(xl[1], sqrt(u[pij_idx+4])))
        x[2] = min(xu[2], max(xl[2], sqrt(u[pij_idx+5])))
        x[3] = min(xu[3], max(xl[3], u[pij_idx+6]))
        x[4] = min(xu[4], max(xl[4], u[pij_idx+7]))
        x[5] = min(xu[5], max(xl[5], -(u[pij_idx]^2 + u[pij_idx+1]^2)))
        x[6] = min(xu[6], max(xl[6], -(u[pij_idx+2]^2 + u[pij_idx+3]^2)))

        param[1,I] = l[pij_idx]
        param[2,I] = l[pij_idx+1]
        param[3,I] = l[pij_idx+2]
        param[4,I] = l[pij_idx+3]
        param[5,I] = l[pij_idx+4]
        param[6,I] = l[pij_idx+5]
        param[7,I] = l[pij_idx+6]
        param[8,I] = l[pij_idx+7]
        param[9,I] = rho[pij_idx]
        param[10,I] = rho[pij_idx+1]
        param[11,I] = rho[pij_idx+2]
        param[12,I] = rho[pij_idx+3]
        param[13,I] = rho[pij_idx+4]
        param[14,I] = rho[pij_idx+5]
        param[15,I] = rho[pij_idx+6]
        param[16,I] = rho[pij_idx+7]
        param[17,I] = xbar[pij_idx] - z[pij_idx]
        param[18,I] = xbar[pij_idx+1] - z[pij_idx+1]
        param[19,I] = xbar[pij_idx+2] - z[pij_idx+2]
        param[20,I] = xbar[pij_idx+3] - z[pij_idx+3]
        param[21,I] = xbar[pij_idx+4] - z[pij_idx+4]
        param[22,I] = xbar[pij_idx+5] - z[pij_idx+5]
        param[23,I] = xbar[pij_idx+6] - z[pij_idx+6]
        param[24,I] = xbar[pij_idx+7] - z[pij_idx+7]

        if major_iter == 1 #info.inner = 1 (first inner iteration)
            param[27,I] = 10.0 #initial ρ_sij = ρ_sji (let ρ the same for all AL terms)
            mu = 10.0
        else
            mu = param[27,I] #using the existing ρ from last inner iteration (admm iteration)
        end

        #eval new Hessian and gradient Ab
        function eval_f_cb(x)
            f = eval_f_polar_linelimit_kernel_cpu_qpsub(I, x, param, scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return f
        end

        function eval_g_cb(x, g)
            eval_grad_f_polar_linelimit_kernel_cpu_qpsub(I, x, g, param, scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end

        function eval_h_cb(x, mode, rows, cols, _scale, lambda, values)
            eval_h_polar_linelimit_kernel_cpu_qpsub(I, x, mode, rows, cols, lambda, values, param,
                              scale, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI)
            return
        end
        
        #internal parameters eta omega mu to guide ALM convergence
        eta = 1 / mu^0.1
        omega = 1 / mu
        max_feval = 500
        max_minor = 100
        gtol = 1e-6

        nele_hess = 20
        tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
                                     :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
                                     :frtol => 1e-12)
        tron.x .= x
        it = 0
        avg_tron_minor = 0
        terminate = false

        while !terminate
            it += 1

            # Solve the branch problem.
            status = ExaTron.solveProblem(tron)
            x .= tron.x
            avg_tron_minor += tron.minor_iter

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
                    param[25,I] += mu*cviol1
                    param[26,I] += mu*cviol2

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10)
                eta = 1 / mu^0.1
                omega = 1 / mu
                param[27,I] = mu
            end

            if it >= max_auglag #maximum iteration for auglag 
                println("max_auglag reached for line I = ", I, " cnorm = ", cnorm)
                terminate = true
            end
        end
        
        #internal solutions are not recorded
        avg_auglag_it += it
        avg_minor_it += (avg_tron_minor / it) #tron iteration for each auglag iter 
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
        param[27,I] = mu
    end

    return (avg_auglag_it / nline), (avg_minor_it / nline) #average auglag iter and tron iter with nline 
end
