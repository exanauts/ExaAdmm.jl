"""
   Internal Solution Structure for branch

- branch structure from u (8*nline):
    - |p_ij   | q_ij  | p_ji   | q_ji    | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji) |

- branch structure for Exatron (8*nline):
    - | t_ij(linelimit) | t_ji(linelimit) | w_ijR  |  w_ijI   | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)

- branch structure for Exatron (6*nline): eliminate w_ijR, wij_I
    - | t_ij(linelimit) | t_ji(linelimit) | wi(ij) | wj(ji) | thetai(ij) | thetaj(ji)

- Hessian inherited from SQP (6*nline):
    - |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
"""


"""
    eval_A_b_branch_kernel_gpu_qpsub()

- prepare QP parameter of ADMM branch kernel (before ALM)
"""


function eval_A_b_branch_kernel_gpu_qpsub(
    H, l, rho, v, z, A_ipopt, b_ipopt, id_line, shift_idx, supY, line_res)
    tx = threadIdx().x
    if tx == 1
        @inbounds begin
                for i = 1:6
                    for j = 1:6
                        A_ipopt[i, j] = H[6*(id_line - 1) + i, j] +
                                rho[shift_idx]*supY[4*(id_line - 1) + 1,i+2]*supY[4*(id_line - 1) + 1,j+2] +
                                rho[shift_idx + 1]*supY[4*(id_line - 1) + 2,i+2]*supY[4*(id_line - 1) + 2,j+2] +
                                rho[shift_idx + 2]*supY[4*(id_line - 1) + 3,i+2]*supY[4*(id_line - 1) + 3,j+2] +
                                rho[shift_idx + 3]*supY[4*(id_line - 1) + 4,i+2]*supY[4*(id_line - 1) + 4,j+2]
                    end
                end

                for j = 1:6
                    b_ipopt[j] = (l[shift_idx] - rho[shift_idx]*(v[shift_idx] - z[shift_idx] - line_res[1, id_line]))*supY[4*(id_line - 1) + 1,j+2] +
                    (l[shift_idx+1] - rho[shift_idx+1]*(v[shift_idx+1] - z[shift_idx+1] - line_res[2, id_line]))*supY[4*(id_line - 1) + 2,j+2] +
                    (l[shift_idx+2] - rho[shift_idx+2]*(v[shift_idx+2] - z[shift_idx+2] -  line_res[3, id_line]))*supY[4*(id_line - 1) + 3,j+2] +
                    (l[shift_idx+3] - rho[shift_idx+3]*(v[shift_idx+3] - z[shift_idx+3] -  line_res[4, id_line]))*supY[4*(id_line - 1) + 4,j+2]
                end

                b_ipopt[3] += l[shift_idx+4] - rho[shift_idx+4]*(v[shift_idx+4] - z[shift_idx+4])
                b_ipopt[4] += l[shift_idx+5] - rho[shift_idx+5]*(v[shift_idx+5] - z[shift_idx+5])
                b_ipopt[5] += l[shift_idx+6] - rho[shift_idx+6]*(v[shift_idx+6] - z[shift_idx+6])
                b_ipopt[6] += l[shift_idx+7] - rho[shift_idx+7]*(v[shift_idx+7] - z[shift_idx+7])

                A_ipopt[3,3] += rho[shift_idx + 4]  #wi(ij)
                A_ipopt[4,4] += rho[shift_idx + 5]  #wj(ji)
                A_ipopt[5,5] += rho[shift_idx + 6]  #thetai(ij)
                A_ipopt[6,6] += rho[shift_idx + 7]   #thetaj(ji)
        end #@inbounds
    end

    CUDA.sync_threads()
    return #6*6 #not scale
end



"""
    eval_A_b_auglag_branch_kernel_gpu_qpsub_red()

- prepare QP parameter of ADMM branch kernel (after model reduction and ALM)
- Input for Tron GPU
- use mod.membuf
"""

function eval_A_b_auglag_branch_kernel_gpu_qpsub_red(Hbr, bbr, A_aug, Atron, btron, scale,vec_1j,vec_1k,membuf,lineidx, Ctron,dtron,RH_1j,RH_1k)
    tx = threadIdx().x
    if tx == 1
        fill!(A_aug, 0.0)
        fill!(Atron, 0.0)

        @inbounds begin
                    for i = 3:8
                        for j = 3:8
                            A_aug[i,j] = Hbr[i-2,j-2]
                        end
                    end



                    for i=1:8
                        for j=1:8
                            A_aug[i,j] += membuf[5,lineidx]*vec_1j[i]*vec_1j[j]
                            A_aug[i,j] += membuf[5,lineidx]*vec_1k[i]*vec_1k[j]
                        end
                    end


                    for i=1:6
                        for j=1:6
                            c1 = A_aug[1,1]*Ctron[1,j] + A_aug[1,2]*Ctron[2,j] + A_aug[1,3]*Ctron[3,j] + A_aug[1,4]*Ctron[4,j] +
                                A_aug[1,5]*Ctron[5,j] + A_aug[1,6]*Ctron[6,j] + A_aug[1,7]*Ctron[7,j] + A_aug[1,8]*Ctron[8,j]

                            c2 = A_aug[2,1]*Ctron[1,j] + A_aug[2,2]*Ctron[2,j] + A_aug[2,3]*Ctron[3,j] + A_aug[2,4]*Ctron[4,j] +
                                A_aug[2,5]*Ctron[5,j] + A_aug[2,6]*Ctron[6,j] + A_aug[2,7]*Ctron[7,j] + A_aug[2,8]*Ctron[8,j]

                            c3 = A_aug[3,1]*Ctron[1,j] + A_aug[3,2]*Ctron[2,j] + A_aug[3,3]*Ctron[3,j] + A_aug[3,4]*Ctron[4,j] +
                                A_aug[3,5]*Ctron[5,j] + A_aug[3,6]*Ctron[6,j] + A_aug[3,7]*Ctron[7,j] + A_aug[3,8]*Ctron[8,j]

                            c4 = A_aug[4,1]*Ctron[1,j] + A_aug[4,2]*Ctron[2,j] + A_aug[4,3]*Ctron[3,j] + A_aug[4,4]*Ctron[4,j] +
                                A_aug[4,5]*Ctron[5,j] + A_aug[4,6]*Ctron[6,j] + A_aug[4,7]*Ctron[7,j] + A_aug[4,8]*Ctron[8,j]

                            c5 = A_aug[5,1]*Ctron[1,j] + A_aug[5,2]*Ctron[2,j] + A_aug[5,3]*Ctron[3,j] + A_aug[5,4]*Ctron[4,j] +
                                A_aug[5,5]*Ctron[5,j] + A_aug[5,6]*Ctron[6,j] + A_aug[5,7]*Ctron[7,j] + A_aug[5,8]*Ctron[8,j]

                            c6 = A_aug[6,1]*Ctron[1,j] + A_aug[6,2]*Ctron[2,j] + A_aug[6,3]*Ctron[3,j] + A_aug[6,4]*Ctron[4,j] +
                                A_aug[6,5]*Ctron[5,j] + A_aug[6,6]*Ctron[6,j] + A_aug[6,7]*Ctron[7,j] + A_aug[6,8]*Ctron[8,j]

                            c7 = A_aug[7,1]*Ctron[1,j] + A_aug[7,2]*Ctron[2,j] + A_aug[7,3]*Ctron[3,j] + A_aug[7,4]*Ctron[4,j] +
                                A_aug[7,5]*Ctron[5,j] + A_aug[7,6]*Ctron[6,j] + A_aug[7,7]*Ctron[7,j] + A_aug[7,8]*Ctron[8,j]

                            c8 = A_aug[8,1]*Ctron[1,j] + A_aug[8,2]*Ctron[2,j] + A_aug[8,3]*Ctron[3,j] + A_aug[8,4]*Ctron[4,j] +
                                A_aug[8,5]*Ctron[5,j] + A_aug[8,6]*Ctron[6,j] + A_aug[8,7]*Ctron[7,j] + A_aug[8,8]*Ctron[8,j]

                            Atron[i,j] = scale*(Ctron[1,i]*c1 + Ctron[2,i]*c2 + Ctron[3,i]*c3 + Ctron[4,i]*c4 + Ctron[5,i]*c5 + Ctron[6,i]*c6 +
                                            Ctron[7,i]*c7 + Ctron[8,i]*c8)

                        end
                    end

                    b1 = A_aug[1,1]*dtron[1] + A_aug[1,2]*dtron[2] + A_aug[1,3]*dtron[3] + A_aug[1,4]*dtron[4] +
                            A_aug[1,5]*dtron[5] + A_aug[1,6]*dtron[6] + A_aug[1,7]*dtron[7] + A_aug[1,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[1] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[1]

                    b2 = A_aug[2,1]*dtron[1] + A_aug[2,2]*dtron[2] + A_aug[2,3]*dtron[3] + A_aug[2,4]*dtron[4] +
                            A_aug[2,5]*dtron[5] + A_aug[2,6]*dtron[6] + A_aug[2,7]*dtron[7] + A_aug[2,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[2] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[2]

                    b3 = A_aug[3,1]*dtron[1] + A_aug[3,2]*dtron[2] + A_aug[3,3]*dtron[3] + A_aug[3,4]*dtron[4] +
                            A_aug[3,5]*dtron[5] + A_aug[3,6]*dtron[6] + A_aug[3,7]*dtron[7] + A_aug[3,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[3] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[3] + bbr[1]

                    b4 = A_aug[4,1]*dtron[1] + A_aug[4,2]*dtron[2] + A_aug[4,3]*dtron[3] + A_aug[4,4]*dtron[4] +
                            A_aug[4,5]*dtron[5] + A_aug[4,6]*dtron[6] + A_aug[4,7]*dtron[7] + A_aug[4,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[4] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[4] + bbr[2]

                    b5 = A_aug[5,1]*dtron[1] + A_aug[5,2]*dtron[2] + A_aug[5,3]*dtron[3] + A_aug[5,4]*dtron[4] +
                            A_aug[5,5]*dtron[5] + A_aug[5,6]*dtron[6] + A_aug[5,7]*dtron[7] + A_aug[5,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[5] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[5] +bbr[3]

                    b6 = A_aug[6,1]*dtron[1] + A_aug[6,2]*dtron[2] + A_aug[6,3]*dtron[3] + A_aug[6,4]*dtron[4] +
                            A_aug[6,5]*dtron[5] + A_aug[6,6]*dtron[6] + A_aug[6,7]*dtron[7] + A_aug[6,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[6] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[6] + bbr[4]

                    b7 = A_aug[7,1]*dtron[1] + A_aug[7,2]*dtron[2] + A_aug[7,3]*dtron[3] + A_aug[7,4]*dtron[4] +
                            A_aug[7,5]*dtron[5] + A_aug[7,6]*dtron[6] + A_aug[7,7]*dtron[7] + A_aug[7,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[7] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[7] + bbr[5]

                    b8 = A_aug[8,1]*dtron[1] + A_aug[8,2]*dtron[2] + A_aug[8,3]*dtron[3] + A_aug[8,4]*dtron[4] +
                            A_aug[8,5]*dtron[5] + A_aug[8,6]*dtron[6] + A_aug[8,7]*dtron[7] + A_aug[8,8]*dtron[8] +
                            (membuf[3,lineidx] - membuf[5,lineidx]*RH_1j[lineidx])*vec_1j[8] +
                            (membuf[4,lineidx] - membuf[5,lineidx]*RH_1k[lineidx])*vec_1k[8] + bbr[6]


                    for i = 1:6
                            btron[i] = scale*(Ctron[1,i]*b1 + Ctron[2,i]*b2 + Ctron[3,i]*b3 + Ctron[4,i]*b4 + Ctron[5,i]*b5 + Ctron[6,i]*b6 +
                                Ctron[7,i]*b7 + Ctron[8,i]*b8)
                    end



        end #inbounds
    end #tx
    CUDA.sync_threads()
    return
end
