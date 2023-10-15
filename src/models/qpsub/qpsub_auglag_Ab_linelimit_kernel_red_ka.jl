"""
    auglag_linelimit_qpsub()

- for certain line (i,j), update sol.u[pij_idx]
- use Exatron, eval_A_b_branch_kernel_gpu_qpsub, eval_A_b_auglag_branch_kernel_gpu_qpsub_red, tron_gpu_test
- LANCELOT ALM algorithm
- with elimination of w_ijR and w_ijI (v2 in overleaf)
- with multiplier output
"""


"""
   Internal Solution Structure for branch

- branch structure from u (8*nline):
    - |p_ij   | q_ij  | p_ji   | q_ji    | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji) |

- branch structure for Exatron (8*nline):
    - | t_ij(linelimit) | t_ji(linelimit) | w_ijR  |  w_ijI   | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)

- branch structure for Exatron (6*nline): eliminate w_ijR, wij_I
    - | t_ij(linelimit) | t_ji(linelimit) | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)

- Hessian inherited from SQP ie sqp_line (6*nline):
    - |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
"""


@kernel function auglag_linelimit_qpsub_ka(Hs, l_curr, rho, u_curr, v_curr, z_curr, YffR, YffI,
    YftR, YftI, YttR, YttI, YtfR, YtfI, inner, max_auglag, mu_max, scale, lqp, uqp, sqp_line,
    membuf, LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k, RH_1k, lambda, line_start, nline, supY, line_res)

    I = @index(Group, Linear)
    J = @index(Local, Linear)
    lineidx = I
    tx = J
    shift_idx = line_start + 8*(lineidx-1)


    n = 6

    x = @localmem Float64 (n,) #memory allocation
    xl = @localmem Float64 (n,)
    xu = @localmem Float64 (n,)
    trg = @localmem Float64 (n,)
    Hbr = @localmem Float64 (n,n)
    bbr = @localmem Float64 (n,)
    vec_1j = @localmem Float64 (8,)
    vec_1k = @localmem Float64 (8,)
    Ctron = @localmem Float64 (8,6)
    dtron = @localmem Float64 (8,)
    A_aug = @localmem Float64 (8,8)
    Atron = @localmem Float64 (6,6)
    btron = @localmem Float64 (6,)

    fill!(Ctron,0.0)
    fill!(A_aug, 0.0)
    fill!(Atron, 0.0)
    CUDA.sync_threads()

    #initialization: variable wrt branch structure wrt Exatron
    x[1] = 0.0
    x[2] = 0.0
    x[3] = sqp_line[3,lineidx]
    x[4] = sqp_line[4,lineidx]
    x[5] = sqp_line[5,lineidx]
    x[6] = sqp_line[6,lineidx]

    xl[1] = 0.0
    xl[2] = 0.0
    xl[3] = lqp[lineidx,3]
    xl[4] = lqp[lineidx,4]
    xl[5] = lqp[lineidx,5]
    xl[6] = lqp[lineidx,6]

    xu[1] = 200000.0
    xu[2] = 200000.0
    xu[3] = uqp[lineidx,3]
    xu[4] = uqp[lineidx,4]
    xu[5] = uqp[lineidx,5]
    xu[6] = uqp[lineidx,6]

    trg[1] = 0.0
    trg[2] = 0.0
    trg[3] = 0.0
    trg[4] = 0.0
    trg[5] = 0.0
    trg[6] = 0.0

    eval_A_b_branch_kernel_gpu_qpsub(
        Hs, l_curr, rho, v_curr, z_curr, Hbr, bbr, lineidx, shift_idx, supY, line_res,
        I, J
    )

    vec_1j[1] = 1 + LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,1] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,1]
    vec_1j[2] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,2] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,2]
    vec_1j[3] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,3] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,3]
    vec_1j[4] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,4] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,4]
    vec_1j[5] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,5] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,5]
    vec_1j[6] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,6] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,6]
    vec_1j[7] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,7] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,7]
    vec_1j[8] = LH_1j[lineidx,1]*supY[4*(lineidx - 1)+1,8] + LH_1j[lineidx,2]*supY[4*(lineidx - 1)+2,8]

    vec_1k[1] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,1] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,1]
    vec_1k[2] = 1 + LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,2] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,2]
    vec_1k[3] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,3] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,3]
    vec_1k[4] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,4] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,4]
    vec_1k[5] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,5] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,5]
    vec_1k[6] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,6] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,6]
    vec_1k[7] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,7] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,7]
    vec_1k[8] = LH_1k[lineidx,1]*supY[4*(lineidx - 1)+3,8] + LH_1k[lineidx,2]*supY[4*(lineidx - 1)+4,8]


    prod = LH_1h[lineidx,1]*LH_1i[lineidx,2]-LH_1h[lineidx,2]*LH_1i[lineidx,1]
    inv11 =  LH_1i[lineidx,2]/prod
    inv12 = -LH_1h[lineidx,2]/prod
    inv21 = -LH_1i[lineidx,1]/prod
    inv22 = LH_1h[lineidx,1]/prod
    Ctron[1,1] = 1.0
    Ctron[2,2] = 1.0
    Ctron[3,1] = 0.0
    Ctron[3,2] = 0.0
    Ctron[3,3] = -inv11*LH_1h[lineidx,3]
    Ctron[3,4] = -inv11*LH_1h[lineidx,4]
    Ctron[3,5] = -inv12*LH_1i[lineidx,3]
    Ctron[3,6] = -inv12*LH_1i[lineidx,4]
    Ctron[4,1] = 0.0
    Ctron[4,2] = 0.0
    Ctron[4,3] = -inv21*LH_1h[lineidx,3]
    Ctron[4,4] = -inv21*LH_1h[lineidx,4]
    Ctron[4,5] = -inv22*LH_1i[lineidx,3]
    Ctron[4,6] = -inv22*LH_1i[lineidx,4]
    Ctron[5,3] = 1.0
    Ctron[6,4] = 1.0
    Ctron[7,5] = 1.0
    Ctron[8,6] = 1.0


    dtron[1] = 0.0
    dtron[2] = 0.0
    dtron[3] = inv11*RH_1h[lineidx] + inv12*RH_1i[lineidx]
    dtron[4] = inv21*RH_1h[lineidx]  + inv22*RH_1i[lineidx]
    dtron[5] = 0.0
    dtron[6] = 0.0
    dtron[7] = 0.0
    dtron[8] = 0.0


    # initialization on penalty
    if inner == 1 #info.inner = 1 (first inner iteration)
        membuf[5,lineidx] = 10.0 #reset ρ_1h = ρ_1i = ρ_1j = ρ_1k (let ρ the same for all AL terms)
        mu = 10.0 # set mu = initial ρ_*
    else
        mu = membuf[5,lineidx] #inherit mu = ρ from the previous inner iteration
    end



    @synchronize
    # set internal parameters eta omega mu to guide ALM convergence
    eta = 1 / mu^0.1


    it = 0 #iteration count
    terminate = false #termination status





    while !terminate
        it += 1

        eval_A_b_auglag_branch_kernel_gpu_qpsub_red(
            Hbr, bbr, A_aug, Atron, btron, scale,vec_1j,vec_1k,membuf,lineidx, Ctron,dtron,RH_1j,RH_1k,
            I, J
        )

        status, minor_iter = tron_gpu_test(n,Atron,btron,x,xl,xu, I, J)

        sqp0 = Ctron[1,1] * x[1] + Ctron[1,2]*x[2] + Ctron[1,3]*x[3] + Ctron[1,4]*x[4] +
                Ctron[1,5]*x[5] + Ctron[1,6]*x[6] + dtron[1]
        sqp1 = Ctron[2,1] * x[1] + Ctron[2,2]*x[2] + Ctron[2,3]*x[3] + Ctron[2,4]*x[4] +
                Ctron[2,5]*x[5] + Ctron[2,6]*x[6] + dtron[2]

        sqp_line[1,lineidx] = Ctron[3,1] * x[1] + Ctron[3,2]*x[2] + Ctron[3,3]*x[3] + Ctron[3,4]*x[4] +
                                Ctron[3,5]*x[5] + Ctron[3,6]*x[6] + dtron[3]
        sqp_line[2,lineidx] = Ctron[4,1] * x[1] + Ctron[4,2]*x[2] + Ctron[4,3]*x[3] + Ctron[4,4]*x[4] +
                                Ctron[4,5]*x[5] + Ctron[4,6]*x[6] + dtron[4]
        sqp_line[3,lineidx] = Ctron[5,1] * x[1] + Ctron[5,2]*x[2] + Ctron[5,3]*x[3] + Ctron[5,4]*x[4] +
                                Ctron[5,5]*x[5] + Ctron[5,6]*x[6] + dtron[5]
        sqp_line[4,lineidx] = Ctron[6,1] * x[1] + Ctron[6,2]*x[2] + Ctron[6,3]*x[3] + Ctron[6,4]*x[4] +
                                Ctron[6,5]*x[5] + Ctron[6,6]*x[6] + dtron[6]
        sqp_line[5,lineidx] = Ctron[7,1] * x[1] + Ctron[7,2]*x[2] + Ctron[7,3]*x[3] + Ctron[7,4]*x[4] +
                                Ctron[7,5]*x[5] + Ctron[7,6]*x[6] + dtron[7]
        sqp_line[6,lineidx] = Ctron[8,1] * x[1] + Ctron[8,2]*x[2] + Ctron[8,3]*x[3] + Ctron[8,4]*x[4] +
                                Ctron[8,5]*x[5] + Ctron[8,6]*x[6] + dtron[8]



        cviol3 = vec_1j[1]*sqp0 + vec_1j[2]*sqp1 + vec_1j[3]*sqp_line[1,lineidx] + vec_1j[4]*sqp_line[2,lineidx] + vec_1j[5]*sqp_line[3,lineidx] +
                    vec_1j[6]*sqp_line[4,lineidx] + vec_1j[7]*sqp_line[5,lineidx] + vec_1j[8]*sqp_line[6,lineidx] - RH_1j[lineidx]

        cviol4 = vec_1k[1]*sqp0 + vec_1k[2]*sqp1 + vec_1k[3]*sqp_line[1,lineidx] + vec_1k[4]*sqp_line[2,lineidx] + vec_1k[5]*sqp_line[3,lineidx] +
                    vec_1k[6]*sqp_line[4,lineidx] + vec_1k[7]*sqp_line[5,lineidx] + vec_1k[8]*sqp_line[6,lineidx] - RH_1k[lineidx]

        cnorm = max(abs(cviol3), abs(cviol4)) #worst violation

        if cnorm <= eta
            if cnorm <= 1e-6
                terminate = true
            else
                if tx == 1
                    membuf[3,lineidx] += mu*cviol3 #λ_1j
                    membuf[4,lineidx] += mu*cviol4 #λ_1k
                end
                eta = eta / mu^0.9

            end
        else
            mu = min(mu_max, mu*10) #increase penalty
            eta = 1 / mu^0.1
            membuf[5,lineidx] = mu #save penalty for current inner iteration
        end

        if it >= max_auglag #maximum iteration for auglag
            if tx == 1
                # TODO printing
                # @cuprintln("max iteration reach at block = ",lineidx, "and threads = ",tx)
            end
            terminate = true
        end

        @synchronize()
    end #end while ALM



    # #save variables
    u_curr[shift_idx] = supY[4*(lineidx - 1) + 1,3]*sqp_line[1,lineidx] + supY[4*(lineidx - 1) + 1,4]*sqp_line[2,lineidx] + supY[4*(lineidx - 1) + 1,5]*sqp_line[3,lineidx] +
                    supY[4*(lineidx - 1) + 1,6]*sqp_line[4,lineidx] + supY[4*(lineidx - 1) + 1,7]*sqp_line[5,lineidx] + supY[4*(lineidx - 1) + 1,8]*sqp_line[6,lineidx] + line_res[1,lineidx]

    u_curr[shift_idx + 1] = supY[4*(lineidx - 1) + 2,3]*sqp_line[1,lineidx] + supY[4*(lineidx - 1) + 2,4]*sqp_line[2,lineidx] + supY[4*(lineidx - 1) + 2,5]*sqp_line[3,lineidx] +
                    supY[4*(lineidx - 1) + 2,6]*sqp_line[4,lineidx] + supY[4*(lineidx - 1) + 2,7]*sqp_line[5,lineidx] + supY[4*(lineidx - 1) + 2,8]*sqp_line[6,lineidx] + line_res[2,lineidx]

    u_curr[shift_idx + 2] = supY[4*(lineidx - 1) + 3,3]*sqp_line[1,lineidx] + supY[4*(lineidx - 1) + 3,4]*sqp_line[2,lineidx] + supY[4*(lineidx - 1) + 3,5]*sqp_line[3,lineidx] +
                    supY[4*(lineidx - 1) + 3,6]*sqp_line[4,lineidx] + supY[4*(lineidx - 1) + 3,7]*sqp_line[5,lineidx] + supY[4*(lineidx - 1) + 3,8]*sqp_line[6,lineidx] + line_res[3,lineidx]

    u_curr[shift_idx + 3] = supY[4*(lineidx - 1) + 4,3]*sqp_line[1,lineidx] + supY[4*(lineidx - 1) + 4,4]*sqp_line[2,lineidx] + supY[4*(lineidx - 1) + 4,5]*sqp_line[3,lineidx] +
                    supY[4*(lineidx - 1) + 4,6]*sqp_line[4,lineidx] + supY[4*(lineidx - 1) + 4,7]*sqp_line[5,lineidx] + supY[4*(lineidx - 1) + 4,8]*sqp_line[6,lineidx] + line_res[4,lineidx]

    u_curr[shift_idx + 4] = x[3] #wi
    u_curr[shift_idx + 5] = x[4] #wj
    u_curr[shift_idx + 6] = x[5] #thetai
    u_curr[shift_idx + 7] = x[6] #thetaj


    #get multiplier
    tmpH11 = inv11
    tmpH12 = inv21
    tmpH21 = inv12
    tmpH22 = inv22

    tmp14i_1 = 2*u_curr[shift_idx]*YftR[lineidx] + 2*u_curr[shift_idx + 1]*(-YftI[lineidx])
    tmp14i_2 = 2*u_curr[shift_idx]*YftI[lineidx] + 2*u_curr[shift_idx + 1]*YftR[lineidx]

    tmp14h_1 = 2*u_curr[shift_idx + 2]*YtfR[lineidx] + 2*u_curr[shift_idx + 3]*(-YtfI[lineidx])
    tmp14h_2 = 2*u_curr[shift_idx + 2]*(-YtfI[lineidx]) + 2*u_curr[shift_idx + 3]*(-YtfR[lineidx])

    trg[1] = Atron[1,1] * x[1] + Atron[1,2]*x[2] + Atron[1,3]*x[3] + Atron[1,4]*x[4] +
            Atron[1,5]*x[5] + Atron[1,6]*x[6] + btron[1]

    trg[2] = Atron[2,1] * x[1] + Atron[2,2]*x[2] + Atron[2,3]*x[3] + Atron[2,4]*x[4] +
            Atron[2,5]*x[5] + Atron[2,6]*x[6] + btron[2]

    lambda[3,lineidx] = -abs(trg[1])
    lambda[4,lineidx] = -abs(trg[2])

    rhs_1 = trg[1]*tmp14i_1 + trg[2]*tmp14h_1 + Hbr[1,1]*sqp_line[1,lineidx] + Hbr[1,2]*sqp_line[2,lineidx] +
                Hbr[1,3]*sqp_line[3,lineidx] + Hbr[1,4]*sqp_line[4,lineidx] + Hbr[1,5]*sqp_line[5,lineidx] +
                Hbr[1,6]*sqp_line[6,lineidx] + bbr[1]

    rhs_2 = trg[1]*tmp14i_2 + trg[2]*tmp14h_2 + Hbr[2,1]*sqp_line[1,lineidx] + Hbr[2,2]*sqp_line[2,lineidx] +
                Hbr[2,3]*sqp_line[3,lineidx] + Hbr[2,4]*sqp_line[4,lineidx] + Hbr[2,5]*sqp_line[5,lineidx] +
                Hbr[2,6]*sqp_line[6,lineidx] + bbr[2]

    lambda[1,lineidx] = -tmpH11*rhs_1 - tmpH12*rhs_2
    lambda[2,lineidx] = -tmpH21*rhs_1 - tmpH22*rhs_2

    @synchronize
end
