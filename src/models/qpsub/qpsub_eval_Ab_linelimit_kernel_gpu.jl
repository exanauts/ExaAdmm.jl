"""
    eval_A_*(), eval_b_*()

- prepare call backs for build_QP_DS and IPOPT benchmark (solve branch kernel directly)
- use mod.membuf (see model.jl)
- TODO: move A_ipopt, b_ipopt to SQOPF for better performance 
"""


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



function eval_A_b_branch_kernel_gpu_qpsub(
    H, l, rho, v, z,
    _YffR, _YffI,
    _YftR, _YftI,
    _YttR, _YttI,
    _YtfR, _YtfI, A_ipopt, b_ipopt, id_line, shift_idx, supY)

    # YffR = _YffR[id_line]; YffI = _YffI[id_line]
    # YftR = _YftR[id_line]; YftI = _YftI[id_line]
    # YttR = _YttR[id_line]; YttI = _YttI[id_line]
    # YtfR = _YtfR[id_line]; YtfI = _YtfI[id_line]
    
    
    #linear transform pij qij pji qji wrt Hessian inherited structure 
    # supY = [YftR YftI YffR 0 0 0;
    # -YftI YftR -YffI 0 0 0;
    # YtfR -YtfI 0 YttR 0 0;
    # -YtfI -YtfR 0 -YttI 0 0]
    # H_new = H
    
    @inbounds begin
            # println(H)
            # H_new .+= rho[1]*supY[1,:]*transpose(supY[1,:]) #pij
            # H_new .+= rho[2]*supY[2,:]*transpose(supY[2,:]) #qij
            # H_new .+= rho[3]*supY[3,:]*transpose(supY[3,:]) #pji
            # H_new .+= rho[4]*supY[4,:]*transpose(supY[4,:]) #qji

            for i = 1:6
                for j = 1:6
                    A_ipopt[6*(id_line - 1) + i, j] = H[6*(id_line - 1) + i, j] 
                            + rho[shift_idx]*supY[1,i]*supY[4*(id_line - 1) + 1,j+2]
                            + rho[shift_idx + 1]*supY[2,i]*supY[4*(id_line - 1) + 2,j+2]
                            + rho[shift_idx + 2]*supY[3,i]*supY[4*(id_line - 1) + 3,j+2]
                            + rho[shift_idx + 3]*supY[4,i]*supY[4*(id_line - 1) + 4,j+2]  
                end
            end

            for j = 1:6
                       # b = zeros(6)
                            # b .+= (l[1] - rho[1]*(v[1]-z[1])) * supY[1,:] #pij
                            # b .+=  (l[2] - rho[2]*(v[2]-z[2])) * supY[2,:] #qij
                            # b .+= (l[3] - rho[3]*(v[3]-z[3])) * supY[3,:] #pji
                            # b .+=  (l[4] - rho[4]*(v[4]-z[4])) * supY[4,:] #qji
                            # b[3] += (l[5] - rho[5]*(v[5]-z[5])) #wi(ij)
                            # b[4] += (l[6] - rho[6]*(v[6]-z[6])) #wj(ji)
                            # b[5] += (l[7] - rho[7]*(v[7]-z[7])) #thetai(ij)
                            # b[6] += (l[8] - rho[8]*(v[8]-z[8])) #thetaj(ji) 
                    
                            b_ipopt[j,id_line] = (l[shift_idx] - rho[shift_idx]*(v[shift_idx] - z[shift_idx]))*supY[4*(id_line - 1) + 1,j+2] + 
                            (l[shift_idx+1] - rho[shift_idx+1]*(v[shift_idx+1] - z[shift_idx+1]))*supY[4*(id_line - 1) + 2,j+2] +
                            (l[shift_idx+2] - rho[shift_idx+2]*(v[shift_idx+2] - z[shift_idx+2]))*supY[4*(id_line - 1) + 3,j+2] +
                            (l[shift_idx+3] - rho[shift_idx+3]*(v[shift_idx+3] - z[shift_idx+3]))*supY[4*(id_line - 1) + 4,j+2]  
            end

            b_ipopt[3, id_line] += l[shift_idx+4] - rho[shift_idx+4]*(v[shift_idx+4] - z[shift_idx+4])
            b_ipopt[4, id_line] += l[shift_idx+5] - rho[shift_idx+5]*(v[shift_idx+5] - z[shift_idx+5])
            b_ipopt[5, id_line] += l[shift_idx+6] - rho[shift_idx+6]*(v[shift_idx+6] - z[shift_idx+6])
            b_ipopt[6, id_line] += l[shift_idx+7] - rho[shift_idx+7]*(v[shift_idx+7] - z[shift_idx+7])
            # println(H)
            # H_new[3,3] += rho[5] #wi(ij) 
            # H_new[4,4] += rho[6] #wj(ji) 
            # H_new[5,5] += rho[7] #thetai(ij)
            # H_new[6,6] += rho[8] #thetaj(ji)
            A_ipopt[6*(id_line - 1) + 3,3] = rho[shift_idx + 4] + H[6*(id_line - 1) + 3,3] #wi(ij) 
            A_ipopt[6*(id_line - 1) + 4,4] = rho[shift_idx + 5] + H[6*(id_line - 1) + 4,4] #wj(ji) 
            A_ipopt[6*(id_line - 1) + 5,5] = rho[shift_idx + 6] + H[6*(id_line - 1) + 5,5] #thetai(ij)
            A_ipopt[6*(id_line - 1) + 6,6] = rho[shift_idx + 7] + H[6*(id_line - 1) + 6,6]  #thetaj(ji)
    end #@inbounds 
    
    #! H may not be perfectly symmetric 
    CUDA.sync_threads()
    return #6*6 #not scale
end


