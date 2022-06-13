"""
    eval_f*(), eval_g*(), eval_h*()

- prepare call backs for ExaTron
- ONLY for benchmark  
"""


"""
   Internal Solution Structure for branch

- branch structure from u 8*nline:   
    - |p_ij   | q_ij  | p_ji   | q_ji    | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji) |
- branch structure for Exatron (8*nline):  
    - | t_ij(linelimit) | t_ji(linelimit) | w_ijR  |  w_ijI   | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)
- Hessian inherited from SQP (6*nline):   
    - |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|   
"""

function eval_f_polar_linelimit_kernel_cpu_qpsub(
    I::Int, x, param, scale::Float64,
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64)

    f = 0.0

    @inbounds begin
        vi_vj_cos = x[1]*x[2]*cos(x[3] - x[4])
        vi_vj_sin = x[1]*x[2]*sin(x[3] - x[4])
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        # from ADMM 
        f += param[1,I]*pij
        f += param[2,I]*qij
        f += param[3,I]*pji
        f += param[4,I]*qji
        f += param[5,I]*x[1]^2
        f += param[6,I]*x[2]^2
        f += param[7,I]*x[3]
        f += param[8,I]*x[4]
       
        f += 0.5*(param[9,I]*(pij - param[17,I])^2)
        f += 0.5*(param[10,I]*(qij - param[18,I])^2)
        f += 0.5*(param[11,I]*(pji - param[19,I])^2)
        f += 0.5*(param[12,I]*(qji - param[20,I])^2)
        f += 0.5*(param[13,I]*(x[1]^2 - param[21,I])^2)
        f += 0.5*(param[14,I]*(x[2]^2 - param[22,I])^2)
        f += 0.5*(param[15,I]*(x[3] - param[23,I])^2)
        f += 0.5*(param[16,I]*(x[4] - param[24,I])^2)

        # Line limit terms
        f += param[25,I]*(pij^2 + qij^2 + x[5]) #λ_sij
        f += param[26,I]*(pji^2 + qji^2 + x[6]) #λ_sji
        f += 0.5*(param[27,I]*(pij^2 + qij^2 + x[5])^2) #ρ_sij
        f += 0.5*(param[27,I]*(pji^2 + qji^2 + x[6])^2) #ρ_sji = ̢ρ_sji (same)
    end

    f *= scale

    return f
end

function eval_grad_f_polar_linelimit_kernel_cpu_qpsub(
    I::Int, x, g, param, scale::Float64,
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64)

    @inbounds begin
        cos_ij = cos(x[3] - x[4])
        sin_ij = sin(x[3] - x[4])
        vi_vj_cos = x[1]*x[2]*cos_ij
        vi_vj_sin = x[1]*x[2]*sin_ij
        pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
        qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
        pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
        qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin

        # Common terms for line limits
        ij_sqsum = (pij^2 + qij^2) + x[5]
        ji_sqsum = (pji^2 + qji^2) + x[6]

        # Derivative with respect to vi.
        dpij_dx = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
        dqij_dx = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
        dpji_dx = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
        dqji_dx = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

        g1 = param[1,I]*(dpij_dx)
        g1 += param[2,I]*(dqij_dx)
        g1 += param[3,I]*(dpji_dx)
        g1 += param[4,I]*(dqji_dx)
        g1 += param[5,I]*(2*x[1])
        g1 += param[9,I]*(pij - param[17,I])*dpij_dx
        g1 += param[10,I]*(qij - param[18,I])*dqij_dx
        g1 += param[11,I]*(pji - param[19,I])*dpji_dx
        g1 += param[12,I]*(qji - param[20,I])*dqji_dx
        g1 += param[13,I]*(x[1]^2 - param[21,I])*(2*x[1])

        # Line limit terms
        g1 += param[25,I]*(2*pij*dpij_dx + 2*qij*dqij_dx)
        g1 += param[26,I]*(2*pji*dpji_dx + 2*qji*dqji_dx)
        g1 += param[27,I]*(ij_sqsum*(2*pij*dpij_dx + 2*qij*dqij_dx))
        g1 += param[27,I]*(ji_sqsum*(2*pji*dpji_dx + 2*qji*dqji_dx))

        # Derivative with respect to vj.
        dpij_dx = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
        dqij_dx = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
        dpji_dx = (2*YttR*x[2] + YtfR*x[1]*cos_ij) - YtfI*x[1]*sin_ij
        dqji_dx = (-2*YttI*x[2] - YtfI*x[1]*cos_ij) - YtfR*x[1]*sin_ij

        g2 = param[1,I]*(dpij_dx)
        g2 += param[2,I]*(dqij_dx)
        g2 += param[3,I]*(dpji_dx)
        g2 += param[4,I]*(dqji_dx)
        g2 += param[6,I]*(2*x[2])
        g2 += param[9,I]*(pij - param[17,I])*dpij_dx
        g2 += param[10,I]*(qij - param[18,I])*dqij_dx
        g2 += param[11,I]*(pji - param[19,I])*dpji_dx
        g2 += param[12,I]*(qji - param[20,I])*dqji_dx
        g2 += param[14,I]*(x[2]^2 - param[22,I])*(2*x[2])

        # Line limit terms
        g2 += param[25,I]*(2*pij*dpij_dx + 2*qij*dqij_dx)
        g2 += param[26,I]*(2*pji*dpji_dx + 2*qji*dqji_dx)
        g2 += param[27,I]*(ij_sqsum*(2*pij*dpij_dx + 2*qij*dqij_dx))
        g2 += param[27,I]*(ji_sqsum*(2*pji*dpji_dx + 2*qji*dqji_dx))

        # Derivative with respect to ti.
        dpij_dx = -YftR*vi_vj_sin + YftI*vi_vj_cos
        dqij_dx = YftI*vi_vj_sin + YftR*vi_vj_cos
        dpji_dx = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
        dqji_dx = YtfI*vi_vj_sin - YtfR*vi_vj_cos

        g3 = param[1,I]*(dpij_dx)
        g3 += param[2,I]*(dqij_dx)
        g3 += param[3,I]*(dpji_dx)
        g3 += param[4,I]*(dqji_dx)
        g3 += param[7,I]
        g3 += param[9,I]*(pij - param[17,I])*dpij_dx
        g3 += param[10,I]*(qij - param[18,I])*dqij_dx
        g3 += param[11,I]*(pji - param[19,I])*dpji_dx
        g3 += param[12,I]*(qji - param[20,I])*dqji_dx
        g3 += param[15,I]*(x[3] - param[23,I])

        # Line limit terms
        g3 += param[25,I]*(2*pij*dpij_dx + 2*qij*dqij_dx)
        g3 += param[26,I]*(2*pji*dpji_dx + 2*qji*dqji_dx)
        g3 += param[27,I]*(ij_sqsum*(2*pij*dpij_dx + 2*qij*dqij_dx))
        g3 += param[27,I]*(ji_sqsum*(2*pji*dpji_dx + 2*qji*dqji_dx))

        # Derivative with respect to tj.

        g4 = param[1,I]*(-dpij_dx)
        g4 += param[2,I]*(-dqij_dx)
        g4 += param[3,I]*(-dpji_dx)
        g4 += param[4,I]*(-dqji_dx)
        g4 += param[8,I]
        g4 += param[9,I]*(pij - param[17,I])*(-dpij_dx)
        g4 += param[10,I]*(qij - param[18,I])*(-dqij_dx)
        g4 += param[11,I]*(pji - param[19,I])*(-dpji_dx)
        g4 += param[12,I]*(qji - param[20,I])*(-dqji_dx)
        g4 += param[16,I]*(x[4] - param[24,I])

        # Line limit terms
        g4 += param[25,I]*(2*pij*(-dpij_dx) + 2*qij*(-dqij_dx))
        g4 += param[26,I]*(2*pji*(-dpji_dx) + 2*qji*(-dqji_dx))
        g4 += param[27,I]*(ij_sqsum*(2*pij*(-dpij_dx) + 2*qij*(-dqij_dx)))
        g4 += param[27,I]*(ji_sqsum*(2*pji*(-dpji_dx) + 2*qji*(-dqji_dx)))

        # Derivative with respect to sij.
        g5 = param[25,I] + param[27,I]*ij_sqsum

        # Derivative with respect to sji.
        g6 = param[26,I] + param[27,I]*ji_sqsum

        g[1] = scale*g1
        g[2] = scale*g2
        g[3] = scale*g3
        g[4] = scale*g4
        g[5] = scale*g5
        g[6] = scale*g6
    end

    return
end

function eval_h_polar_linelimit_kernel_cpu_qpsub(
    I::Int, x, mode, rows, cols, lambda, values, param, scale::Float64,
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64)

    @inbounds begin
        if mode == :Structure
            nz = 1
            rows[nz] = 1; cols[nz] = 1; nz += 1
            rows[nz] = 2; cols[nz] = 1; nz += 1
            rows[nz] = 3; cols[nz] = 1; nz += 1
            rows[nz] = 4; cols[nz] = 1; nz += 1
            rows[nz] = 5; cols[nz] = 1; nz += 1
            rows[nz] = 6; cols[nz] = 1; nz += 1

            rows[nz] = 2; cols[nz] = 2; nz += 1
            rows[nz] = 3; cols[nz] = 2; nz += 1
            rows[nz] = 4; cols[nz] = 2; nz += 1
            rows[nz] = 5; cols[nz] = 2; nz += 1
            rows[nz] = 6; cols[nz] = 2; nz += 1

            rows[nz] = 3; cols[nz] = 3; nz += 1
            rows[nz] = 4; cols[nz] = 3; nz += 1
            rows[nz] = 5; cols[nz] = 3; nz += 1
            rows[nz] = 6; cols[nz] = 3; nz += 1

            rows[nz] = 4; cols[nz] = 4; nz += 1
            rows[nz] = 5; cols[nz] = 4; nz += 1
            rows[nz] = 6; cols[nz] = 4; nz += 1

            rows[nz] = 5; cols[nz] = 5; nz += 1
            rows[nz] = 6; cols[nz] = 6; nz += 1
        else
            nz = 1

            cos_ij = cos(x[3] - x[4])
            sin_ij = sin(x[3] - x[4])
            vi_vj_cos = x[1]*x[2]*cos_ij
            vi_vj_sin = x[1]*x[2]*sin_ij
            pij = YffR*x[1]^2 + YftR*vi_vj_cos + YftI*vi_vj_sin
            qij = -YffI*x[1]^2 - YftI*vi_vj_cos + YftR*vi_vj_sin
            pji = YttR*x[2]^2 + YtfR*vi_vj_cos - YtfI*vi_vj_sin
            qji = -YttI*x[2]^2 - YtfI*vi_vj_cos - YtfR*vi_vj_sin
            ij_sqsum = pij^2 + qij^2 + x[5]
            ji_sqsum = pji^2 + qji^2 + x[6]

            # d2f_dvidvi

            dpij_dvi = 2*YffR*x[1] + YftR*x[2]*cos_ij + YftI*x[2]*sin_ij
            dqij_dvi = -2*YffI*x[1] - YftI*x[2]*cos_ij + YftR*x[2]*sin_ij
            dpji_dvi = YtfR*x[2]*cos_ij - YtfI*x[2]*sin_ij
            dqji_dvi = -YtfI*x[2]*cos_ij - YtfR*x[2]*sin_ij

            # l_pij * d2pij_dvidvi
            v = param[1,I]*(2*YffR)
            # l_qij * d2qij_dvidvi
            v += param[2,I]*(-2*YffI)
            # l_pji * d2pji_dvidvi = 0
            # l_qji * d2qji_dvidvi = 0
            # l_vi * 2
            v += 2*param[5,I]
            # rho_pij*(dpij_dvi)^2 + rho_pij*(pij - tilde_pij)*d2pij_dvidvi
            v += param[9,I]*(dpij_dvi)^2 + param[9,I]*(pij - param[17,I])*(2*YffR)
            # rho_qij*(dqij_dvi)^2 + rho_qij*(qij - tilde_qij)*d2qij_dvidvi
            v += param[10,I]*(dqij_dvi)^2 + param[10,I]*(qij - param[18,I])*(-2*YffI)
            # rho_pji*(dpji_dvi)^2 + rho_pji*(pji - tilde_pji)*d2pji_dvidvi
            v += param[11,I]*(dpji_dvi)^2
            # rho_qji*(dqji_dvi)^2
            v += param[12,I]*(dqji_dvi)^2
            # (2*rho_vi*vi)*(2*vi) + rho_vi*(vi^2 - tilde_vi)*2
            v += 4*param[13,I]*x[1]^2 + param[13,I]*(x[1]^2 - param[21,I])*2

            # Line limit terms
            d2ij_sqsum_dvdv = 2*(dpij_dvi)^2 + 2*pij*(2*YffR) + 2*(dqij_dvi)^2 + 2*qij*(-2*YffI)
            d2ji_sqsum_dvdv = 2*(dpji_dvi)^2 + 2*(dqji_dvi)^2
            # l_sij*(2*(dpij_dvi)^2 + 2*pij*(d2pij_dvidvi) + 2*(dqij_dvi)^2 + 2*qij*(d2qij_dvidvi))
            v += param[25,I]*d2ij_sqsum_dvdv
            # l_sji*(2*(dpji_dvi)^2 + 2*pji*(d2pji_dvidvi) + 2*(dqji_dvi)^2 + 2*qji*(d2qji_dvidvi))
            v += param[26,I]*d2ji_sqsum_dvdv
            # rho_sij*((d2ij_sqsum_dvidvi)^2 + ij_sqsum*(d2ij_sqsum_dvidvi)))
            v += param[27,I]*((2*pij*dpij_dvi + 2*qij*dqij_dvi)^2 + ij_sqsum*d2ij_sqsum_dvdv)
            # rho_sij*((d2ji_sqsum_dvidvi)^2 + ji_sqsum*(d2ji_sqsum_dvidvi)))
            v += param[27,I]*((2*pji*dpji_dvi + 2*qji*dqji_dvi)^2 + ji_sqsum*d2ji_sqsum_dvdv)
            values[nz] = scale*v
            nz += 1

            # d2f_dvidvj

            dpij_dvj = YftR*x[1]*cos_ij + YftI*x[1]*sin_ij
            dqij_dvj = -YftI*x[1]*cos_ij + YftR*x[1]*sin_ij
            dpji_dvj = 2*YttR*x[2] + YtfR*x[1]*cos_ij - YtfI*x[1]*sin_ij
            dqji_dvj = -2*YttI*x[2] - YtfI*x[1]*cos_ij - YtfR*x[1]*sin_ij

            d2pij_dvidvj = YftR*cos_ij + YftI*sin_ij
            d2qij_dvidvj = -YftI*cos_ij + YftR*sin_ij
            d2pji_dvidvj = YtfR*cos_ij - YtfI*sin_ij
            d2qji_dvidvj = -YtfI*cos_ij - YtfR*sin_ij

            # l_pij * d2pij_dvidvj
            v = param[1,I]*(d2pij_dvidvj)
            # l_qij * d2qij_dvidvj
            v += param[2,I]*(d2qij_dvidvj)
            # l_pji * d2pji_dvidvj
            v += param[3,I]*(d2pji_dvidvj)
            # l_qji * d2qji_dvidvj
            v += param[4,I]*(d2qji_dvidvj)
            # rho_pij*(dpij_dvj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidvj)
            v += param[9,I]*(dpij_dvj)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidvj)
            # rho_qij*(dqij_dvj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidvj)
            v += param[10,I]*(dqij_dvj)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidvj)
            # rho_pji*(dpji_dvj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidvj)
            v += param[11,I]*(dpji_dvj)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidvj)
            # rho_qji*(dqji_dvj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidvj)
            v += param[12,I]*(dqji_dvj)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidvj)

            # Line limit terms
            d2ij_sqsum_dvdv = 2*dpij_dvj*dpij_dvi + 2*pij*d2pij_dvidvj + 2*dqij_dvj*dqij_dvi + 2*qij*d2qij_dvidvj
            d2ji_sqsum_dvdv = 2*dpji_dvj*dpji_dvi + 2*pji*d2pji_dvidvj + 2*dqji_dvj*dqji_dvi + 2*qji*d2qji_dvidvj
            v += param[25,I]*d2ij_sqsum_dvdv
            v += param[26,I]*d2ji_sqsum_dvdv
            v += param[27,I]*((2*pij*dpij_dvj + 2*qij*dqij_dvj)*(2*pij*dpij_dvi + 2*qij*dqij_dvi) + ij_sqsum*(d2ij_sqsum_dvdv))
            v += param[27,I]*((2*pji*dpji_dvj + 2*qji*dqji_dvj)*(2*pji*dpji_dvi + 2*qji*dqji_dvi) + ji_sqsum*(d2ji_sqsum_dvdv))
            values[nz] = scale*v
            nz += 1

            # d2f_dvidti

            dpij_dti = -YftR*vi_vj_sin + YftI*vi_vj_cos
            dqij_dti = YftI*vi_vj_sin + YftR*vi_vj_cos
            dpji_dti = -YtfR*vi_vj_sin - YtfI*vi_vj_cos
            dqji_dti = YtfI*vi_vj_sin - YtfR*vi_vj_cos

            d2pij_dvidti = -YftR*x[2]*sin_ij + YftI*x[2]*cos_ij
            d2qij_dvidti = YftI*x[2]*sin_ij + YftR*x[2]*cos_ij
            d2pji_dvidti = -YtfR*x[2]*sin_ij - YtfI*x[2]*cos_ij
            d2qji_dvidti = YtfI*x[2]*sin_ij - YtfR*x[2]*cos_ij

            # l_pij * d2pij_dvidti
            v = param[1,I]*(d2pij_dvidti)
            # l_qij * d2qij_dvidti
            v += param[2,I]*(d2qij_dvidti)
            # l_pji * d2pji_dvidti
            v += param[3,I]*(d2pji_dvidti)
            # l_qji * d2qji_dvidti
            v += param[4,I]*(d2qji_dvidti)
            # rho_pij*(dpij_dti)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidti)
            v += param[9,I]*(dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(d2pij_dvidti)
            # rho_qij*(dqij_dti)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidti)
            v += param[10,I]*(dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(d2qij_dvidti)
            # rho_pji*(dpji_dti)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidti)
            v += param[11,I]*(dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(d2pji_dvidti)
            # rho_qji*(dqji_dti)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidti)
            v += param[12,I]*(dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(d2qji_dvidti)

            # Line limit terms
            d2ij_sqsum_dvdt = 2*dpij_dti*dpij_dvi + 2*pij*d2pij_dvidti + 2*dqij_dti*dqij_dvi + 2*qij*d2qij_dvidti
            d2ji_sqsum_dvdt = 2*dpji_dti*dpji_dvi + 2*pji*d2pji_dvidti + 2*dqji_dti*dqji_dvi + 2*qji*d2qji_dvidti
            v += param[25,I]*d2ij_sqsum_dvdt
            v += param[26,I]*d2ji_sqsum_dvdt
            v += param[27,I]*((2*pij*dpij_dti + 2*qij*dqij_dti)*(2*pij*dpij_dvi + 2*qij*dqij_dvi) + ij_sqsum*d2ij_sqsum_dvdt)
            v += param[27,I]*((2*pji*dpji_dti + 2*qji*dqji_dti)*(2*pji*dpji_dvi + 2*qji*dqji_dvi) + ji_sqsum*d2ji_sqsum_dvdt)
            values[nz] = scale*v
            nz += 1

            # d2f_dvidtj

            # l_pij * d2pij_dvidtj
            v = param[1,I]*(-d2pij_dvidti)
            # l_qij * d2qij_dvidtj
            v += param[2,I]*(-d2qij_dvidti)
            # l_pji * d2pji_dvidtj
            v += param[3,I]*(-d2pji_dvidti)
            # l_qji * d2qji_dvidtj
            v += param[4,I]*(-d2qji_dvidti)
            # rho_pij*(dpij_dtj)*dpij_dvi + rho_pij*(pij - tilde_pij)*(d2pij_dvidtj)
            v += param[9,I]*(-dpij_dti)*dpij_dvi + param[9,I]*(pij - param[17,I])*(-d2pij_dvidti)
            # rho_qij*(dqij_dtj)*dqij_dvi + rho_qij*(qij - tilde_qij)*(d2qij_dvidtj)
            v += param[10,I]*(-dqij_dti)*dqij_dvi + param[10,I]*(qij - param[18,I])*(-d2qij_dvidti)
            # rho_pji*(dpji_dtj)*dpji_dvi + rho_pji*(pji - tilde_pji)*(d2pji_dvidtj)
            v += param[11,I]*(-dpji_dti)*dpji_dvi + param[11,I]*(pji - param[19,I])*(-d2pji_dvidti)
            # rho_qji*(dqji_dtj)*dqji_dvi + rho_qji*(qji - tilde_qji)*(d2qji_dvidtj)
            v += param[12,I]*(-dqji_dti)*dqji_dvi + param[12,I]*(qji - param[20,I])*(-d2qji_dvidti)

            # Line limit terms
            v += param[25,I]*(-d2ij_sqsum_dvdt)
            v += param[26,I]*(-d2ji_sqsum_dvdt)
            v += param[27,I]*(-(2*pij*dpij_dti + 2*qij*dqij_dti)*(2*pij*dpij_dvi + 2*qij*dqij_dvi) + ij_sqsum*(-d2ij_sqsum_dvdt))
            v += param[27,I]*(-(2*pji*dpji_dti + 2*qji*dqji_dti)*(2*pji*dpji_dvi + 2*qji*dqji_dvi) + ji_sqsum*(-d2ji_sqsum_dvdt))
            values[nz] = scale*v
            nz += 1

            # d2f_dvidsij
            v = param[27,I]*(2*pij*dpij_dvi + 2*qij*dqij_dvi)
            values[nz] = scale*v
            nz += 1

            # d2f_dvidsji
            v = param[27,I]*(2*pji*dpji_dvi + 2*qji*dqji_dvi)
            values[nz] = scale*v
            nz += 1

            # d2f_dvjdvj

            # l_pij * d2pij_dvjdvj = l_qij * d2qij_dvjdvj = 0 since d2pij_dvjdvj = d2qij_dvjdvj = 0
            # l_pji * d2pji_dvjdvj
            v = param[3,I]*(2*YttR)
            # l_qji * d2qji_dvjdvj
            v += param[4,I]*(-2*YttI)
            # l_vj * 2
            v += param[6,I]*2
            # rho_pij*(dpij_dvj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dvjdvj)
            v += param[9,I]*(dpij_dvj)^2
            # rho_qij*(dqij_dvj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dvjdvj)
            v += param[10,I]*(dqij_dvj)^2
            # rho_pji*(dpji_dvj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dvjdvj)
            v += param[11,I]*(dpji_dvj)^2 + param[11,I]*(pji - param[19,I])*(2*YttR)
            # rho_qji*(dqji_dvj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dvjdvj)
            v += param[12,I]*(dqji_dvj)^2 + param[12,I]*(qji - param[20,I])*(-2*YttI)
            # (2*rho_vj*vj)*(2*vj) + rho_vj*(vj^2 - tilde_vj)*2
            v += 4*param[14,I]*x[2]^2 + param[14,I]*(x[2]^2 - param[22,I])*2

            # Line limit terms
            d2ij_sqsum_dvdv = 2*(dpij_dvj)^2 + 2*(dqij_dvj)^2
            d2ji_sqsum_dvdv = 2*(dpji_dvj)^2 + 2*pji*(2*YttR) + 2*(dqji_dvj)^2 + 2*qji*(-2*YttI)
            # l_sij*(2*(dpij_dvi)^2 + 2*pij*(d2pij_dvidvi) + 2*(dqij_dvi)^2 + 2*qij*(d2qij_dvidvi))
            v += param[25,I]*d2ij_sqsum_dvdv
            # l_sji*(2*(dpji_dvi)^2 + 2*pji*(d2pji_dvidvi) + 2*(dqji_dvi)^2 + 2*qji*(d2qji_dvidvi))
            v += param[26,I]*d2ji_sqsum_dvdv
            # rho_sij*((d2ij_sqsum_dvjdvj)^2 + ij_sqsum*(d2ij_sqsum_dvjdvj)))
            v += param[27,I]*((2*pij*dpij_dvj + 2*qij*dqij_dvj)^2 + ij_sqsum*d2ij_sqsum_dvdv)
            # rho_sij*((d2ji_sqsum_dvjdvj)^2 + ji_sqsum*(d2ji_sqsum_dvjdvj)))
            v += param[27,I]*((2*pji*dpji_dvj + 2*qji*dqji_dvj)^2 + ji_sqsum*d2ji_sqsum_dvdv)
            values[nz] = scale*v
            nz += 1

            # d2f_dvjdti

            d2pij_dvjdti = (-YftR*x[1]*sin_ij + YftI*x[1]*cos_ij)
            d2qij_dvjdti = (YftI*x[1]*sin_ij + YftR*x[1]*cos_ij)
            d2pji_dvjdti = (-YtfR*x[1]*sin_ij - YtfI*x[1]*cos_ij)
            d2qji_dvjdti = (YtfI*x[1]*sin_ij - YtfR*x[1]*cos_ij)

            # l_pij * d2pij_dvjdti
            v = param[1,I]*(d2pij_dvjdti)
            # l_qij * d2qij_dvjdti
            v += param[2,I]*(d2qij_dvjdti)
            # l_pji * d2pji_dvjdti
            v += param[3,I]*(d2pji_dvjdti)
            # l_qji * d2qji_dvjdti
            v += param[4,I]*(d2qji_dvjdti)
            # rho_pij*(dpij_dti)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdti)
            v += param[9,I]*(dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*d2pij_dvjdti
            # rho_qij*(dqij_dti)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdti)
            v += param[10,I]*(dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*d2qij_dvjdti
            # rho_pji*(dpji_dti)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdti)
            v += param[11,I]*(dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*d2pji_dvjdti
            # rho_qji*(dqji_dti)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdti)
            v += param[12,I]*(dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*d2qji_dvjdti

            # Line limit terms
            d2ij_sqsum_dvdt = 2*dpij_dti*dpij_dvj + 2*pij*d2pij_dvjdti + 2*dqij_dti*dqij_dvj + 2*qij*d2qij_dvjdti
            d2ji_sqsum_dvdt = 2*dpji_dti*dpji_dvj + 2*pji*d2pji_dvjdti + 2*dqji_dti*dqji_dvj + 2*qji*d2qji_dvjdti
            v += param[25,I]*d2ij_sqsum_dvdt
            v += param[26,I]*d2ji_sqsum_dvdt
            v += param[27,I]*((2*pij*dpij_dti + 2*qij*dqij_dti)*(2*pij*dpij_dvj + 2*qij*dqij_dvj) + ij_sqsum*d2ij_sqsum_dvdt)
            v += param[27,I]*((2*pji*dpji_dti + 2*qji*dqji_dti)*(2*pji*dpji_dvj + 2*qji*dqji_dvj) + ji_sqsum*d2ji_sqsum_dvdt)
            values[nz] = scale*v
            nz += 1

            # d2f_dvjdtj

            # l_pij * d2pij_dvjdtj
            v = param[1,I]*(-d2pij_dvjdti)
            # l_qij * d2qij_dvjdtj
            v += param[2,I]*(-d2qij_dvjdti)
            # l_pji * d2pji_dvjdtj
            v += param[3,I]*(-d2pji_dvjdti)
            # l_qji * d2qji_dvjdtj
            v += param[4,I]*(-d2qji_dvjdti)
            # rho_pij*(dpij_dtj)*dpij_dvj + rho_pij*(pij - tilde_pij)*(d2pij_dvjdtj)
            v += param[9,I]*(-dpij_dti)*dpij_dvj + param[9,I]*(pij - param[17,I])*(-d2pij_dvjdti)
            # rho_qij*(dqij_dtj)*dqij_dvj + rho_qij*(qij - tilde_qij)*(d2qij_dvjdtj)
            v += param[10,I]*(-dqij_dti)*dqij_dvj + param[10,I]*(qij - param[18,I])*(-d2qij_dvjdti)
            # rho_pji*(dpji_dtj)*dpji_dvj + rho_pji*(pji - tilde_pji)*(d2pji_dvjdtj)
            v += param[11,I]*(-dpji_dti)*dpji_dvj + param[11,I]*(pji - param[19,I])*(-d2pji_dvjdti)
            # rho_qji*(dqji_dtj)*dqji_dvj + rho_qji*(qji - tilde_qji)*(d2qji_dvjdtj)
            v += param[12,I]*(-dqji_dti)*dqji_dvj + param[12,I]*(qji - param[20,I])*(-d2qji_dvjdti)

            # Line limits
            v += param[25,I]*(-d2ij_sqsum_dvdt)
            v += param[26,I]*(-d2ji_sqsum_dvdt)
            v += param[27,I]*(-(2*pij*dpij_dti + 2*qij*dqij_dti)*(2*pij*dpij_dvj + 2*qij*dqij_dvj) + ij_sqsum*(-d2ij_sqsum_dvdt))
            v += param[27,I]*(-(2*pji*dpji_dti + 2*qji*dqji_dti)*(2*pji*dpji_dvj + 2*qji*dqji_dvj) + ji_sqsum*(-d2ji_sqsum_dvdt))
            values[nz] = scale*v
            nz += 1

            # d2f_dvjdsij
            v = param[27,I]*(2*pij*dpij_dvj + 2*qij*dqij_dvj)
            values[nz] = scale*v
            nz += 1

            # d2f_dvjdsij
            v = param[27,I]*(2*pji*dpji_dvj + 2*qji*dqji_dvj)
            values[nz] = scale*v
            nz += 1

            # d2f_dtidti

            d2pij_dtidti = (-YftR*vi_vj_cos - YftI*vi_vj_sin)
            d2qij_dtidti = (YftI*vi_vj_cos - YftR*vi_vj_sin)
            d2pji_dtidti = (-YtfR*vi_vj_cos + YtfI*vi_vj_sin)
            d2qji_dtidti = (YtfI*vi_vj_cos + YtfR*vi_vj_sin)

            # l_pij * d2pij_dtidti
            v = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtidti
            v += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtidti
            v += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtidti
            v += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dti)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtidti)
            v += param[9,I]*(dpij_dti)^2 + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dti)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtidti)
            v += param[10,I]*(dqij_dti)^2 + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dti)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtidti)
            v += param[11,I]*(dpji_dti)^2 + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dti)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtidti)
            v += param[12,I]*(dqji_dti)^2 + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_ti
            v += param[15,I]

            # Line limits
            d2ij_sqsum_dtdt = 2*(dpij_dti)^2 + 2*pij*d2pij_dtidti + 2*(dqij_dti)^2 + 2*qij*d2qij_dtidti
            d2ji_sqsum_dtdt = 2*(dpji_dti)^2 + 2*pji*d2pji_dtidti + 2*(dqji_dti)^2 + 2*qji*d2qji_dtidti
            v += param[25,I]*d2ij_sqsum_dtdt
            v += param[26,I]*d2ji_sqsum_dtdt
            v += param[27,I]*((2*pij*dpij_dti + 2*qij*dqij_dti)^2 + ij_sqsum*d2ij_sqsum_dtdt)
            v += param[27,I]*((2*pji*dpji_dti + 2*qji*dqji_dti)^2 + ji_sqsum*d2ji_sqsum_dtdt)
            values[nz] = scale*v
            nz += 1

            # d2f_dtidtj

            # l_pij * d2pij_dtidtj
            v = param[1,I]*(-d2pij_dtidti)
            # l_qij * d2qij_dtidtj
            v += param[2,I]*(-d2qij_dtidti)
            # l_pji * d2pji_dtidtj
            v += param[3,I]*(-d2pji_dtidti)
            # l_qji * d2qji_dtidtj
            v += param[4,I]*(-d2qji_dtidti)
            # rho_pij*(dpij_dtj)*dpij_dti + rho_pij*(pij - tilde_pij)*(d2pij_dtidtj)
            v += param[9,I]*(-dpij_dti^2) + param[9,I]*(pij - param[17,I])*(-d2pij_dtidti)
            # rho_qij*(dqij_dtj)*dqij_dti + rho_qij*(qij - tilde_qij)*(d2qij_dtidtj)
            v += param[10,I]*(-dqij_dti^2) + param[10,I]*(qij - param[18,I])*(-d2qij_dtidti)
            # rho_pji*(dpji_dtj)*dpji_dti + rho_pji*(pji - tilde_pji)*(d2pji_dtidtj)
            v += param[11,I]*(-dpji_dti^2) + param[11,I]*(pji - param[19,I])*(-d2pji_dtidti)
            # rho_qji*(dqji_dtj)*dqji_dti + rho_qji*(qji - tilde_qji)*(d2qji_dtidtj)
            v += param[12,I]*(-dqji_dti^2) + param[12,I]*(qji - param[20,I])*(-d2qji_dtidti)

            # Line limits
            v += param[25,I]*(-d2ij_sqsum_dtdt)
            v += param[26,I]*(-d2ji_sqsum_dtdt)
            v += param[27,I]*(-(2*pij*dpij_dti + 2*qij*dqij_dti)^2 + ij_sqsum*(-d2ij_sqsum_dtdt))
            v += param[27,I]*(-(2*pji*dpji_dti + 2*qji*dqji_dti)^2 + ji_sqsum*(-d2ji_sqsum_dtdt))
            values[nz] = scale*v
            nz += 1

            # d2f_dtidsij
            v = param[27,I]*(2*pij*dpij_dti + 2*qij*dqij_dti)
            values[nz] = scale*v
            nz += 1

            # d2f_dtidsji
            v = param[27,I]*(2*pji*dpji_dti + 2*qji*dqji_dti)
            values[nz] = scale*v
            nz += 1

            # d2f_dtjdtj
            # l_pij * d2pij_dtjdtj
            v = param[1,I]*(d2pij_dtidti)
            # l_qij * d2qij_dtjdtj
            v += param[2,I]*(d2qij_dtidti)
            # l_pji * d2pji_dtjdtj
            v += param[3,I]*(d2pji_dtidti)
            # l_qji * d2qji_dtjdtj
            v += param[4,I]*(d2qji_dtidti)
            # rho_pij*(dpij_dtj)^2 + rho_pij*(pij - tilde_pij)*(d2pij_dtjdtj)
            v += param[9,I]*(dpij_dti^2) + param[9,I]*(pij - param[17,I])*(d2pij_dtidti)
            # rho_qij*(dqij_dtj)^2 + rho_qij*(qij - tilde_qij)*(d2qij_dtjdtj)
            v += param[10,I]*(dqij_dti^2) + param[10,I]*(qij - param[18,I])*(d2qij_dtidti)
            # rho_pji*(dpji_dtj)^2 + rho_pji*(pji - tilde_pji)*(d2pji_dtjdtj)
            v += param[11,I]*(dpji_dti^2) + param[11,I]*(pji - param[19,I])*(d2pji_dtidti)
            # rho_qji*(dqji_dtj)^2 + rho_qji*(qji - tilde_qji)*(d2qji_dtjdtj)
            v += param[12,I]*(dqji_dti^2) + param[12,I]*(qji - param[20,I])*(d2qji_dtidti)
            # rho_tj
            v += param[16,I]

            # Line limits
            v += param[25,I]*d2ij_sqsum_dtdt
            v += param[26,I]*d2ji_sqsum_dtdt
            v += param[27,I]*((2*pij*dpij_dti + 2*qij*dqij_dti)^2 + ij_sqsum*d2ij_sqsum_dtdt)
            v += param[27,I]*((2*pji*dpji_dti + 2*qji*dqji_dti)^2 + ji_sqsum*d2ji_sqsum_dtdt)
            values[nz] = scale*v
            nz += 1

            # d2f_dtjdsij
            v = param[27,I]*(-(2*pij*dpij_dti + 2*qij*dqij_dti))
            values[nz] = scale*v
            nz += 1

            # d2f_dtjdsji
            v = param[27,I]*(-(2*pji*dpji_dti + 2*qji*dqji_dti))
            values[nz] = scale*v
            nz += 1

            # d2f_dsijdsij
            values[nz] = scale*param[27,I]
            nz += 1

            # d2f_dsijdsji = 0

            # d2f_dsjidsji
            values[nz] = scale*param[27,I]
        end
    end

    return
end
