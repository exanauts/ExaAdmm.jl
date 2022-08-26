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
    H, l, rho, v, z, A_ipopt, b_ipopt, id_line, shift_idx, supY, tx)

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
    if tx <= 1
        @inbounds begin
                # println(H)
                # H_new .+= rho[1]*supY[1,:]*transpose(supY[1,:]) #pij
                # H_new .+= rho[2]*supY[2,:]*transpose(supY[2,:]) #qij
                # H_new .+= rho[3]*supY[3,:]*transpose(supY[3,:]) #pji
                # H_new .+= rho[4]*supY[4,:]*transpose(supY[4,:]) #qji

                for i = 1:6
                    for j = 1:6
                        A_ipopt[i, j] = H[6*(id_line - 1) + i, j] +
                                rho[shift_idx]*supY[4*(id_line - 1) + 1,i+2]*supY[4*(id_line - 1) + 1,j+2] +
                                rho[shift_idx + 1]*supY[4*(id_line - 1) + 2,i+2]*supY[4*(id_line - 1) + 2,j+2] +
                                rho[shift_idx + 2]*supY[4*(id_line - 1) + 3,i+2]*supY[4*(id_line - 1) + 3,j+2] +
                                rho[shift_idx + 3]*supY[4*(id_line - 1) + 4,i+2]*supY[4*(id_line - 1) + 4,j+2] 
                        # if id_line == 1
                        #     @cuprintln(H[6*(id_line - 1) + i, j] )   
                        #     @cuprintln(A_ipopt[i,j])   
                        # end      
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
                        
                                b_ipopt[j] = (l[shift_idx] - rho[shift_idx]*(v[shift_idx] - z[shift_idx]))*supY[4*(id_line - 1) + 1,j+2] + 
                                (l[shift_idx+1] - rho[shift_idx+1]*(v[shift_idx+1] - z[shift_idx+1]))*supY[4*(id_line - 1) + 2,j+2] +
                                (l[shift_idx+2] - rho[shift_idx+2]*(v[shift_idx+2] - z[shift_idx+2]))*supY[4*(id_line - 1) + 3,j+2] +
                                (l[shift_idx+3] - rho[shift_idx+3]*(v[shift_idx+3] - z[shift_idx+3]))*supY[4*(id_line - 1) + 4,j+2]  
                end

                b_ipopt[3] += l[shift_idx+4] - rho[shift_idx+4]*(v[shift_idx+4] - z[shift_idx+4])
                b_ipopt[4] += l[shift_idx+5] - rho[shift_idx+5]*(v[shift_idx+5] - z[shift_idx+5])
                b_ipopt[5] += l[shift_idx+6] - rho[shift_idx+6]*(v[shift_idx+6] - z[shift_idx+6])
                b_ipopt[6] += l[shift_idx+7] - rho[shift_idx+7]*(v[shift_idx+7] - z[shift_idx+7])
                # println(H)
                # H_new[3,3] += rho[5] #wi(ij) 
                # H_new[4,4] += rho[6] #wj(ji) 
                # H_new[5,5] += rho[7] #thetai(ij)
                # H_new[6,6] += rho[8] #thetaj(ji)
                A_ipopt[3,3] += rho[shift_idx + 4]  #wi(ij) 
                A_ipopt[4,4] += rho[shift_idx + 5]  #wj(ji) 
                A_ipopt[5,5] += rho[shift_idx + 6]  #thetai(ij)
                A_ipopt[6,6] += rho[shift_idx + 7]   #thetaj(ji)
        end #@inbounds 
    end
    #! H may not be perfectly symmetric 
    CUDA.sync_threads()
    return #6*6 #not scale
end


function eval_A_auglag_branch_kernel_cpu_qpsub_red(Hbr, bbr, A_aug, Atron, btron, scale,vec_1j,vec_1k,membuf,lineidx,tx,Ctron,dtron,RH_1j,RH_1k)
        #     A = zeros(8,8)
        #     A[3:8,3:8] = Hbr
    if tx <= 1
        fill!(A_aug, 0.0)
        fill!(Atron, 0.0)
        
        #     #pij qij pji qji wrt branch structure ExaTron
        #     supY = [0 0 YftR YftI YffR 0 0 0;
        #     0 0 -YftI YftR -YffI 0 0 0;
        #     0 0 YtfR -YtfI 0 YttR 0 0;
        #     0 0 -YtfI -YtfR 0 -YttI 0 0]

        #     #ALM on equality constraint wrt branch structure ExaTron
        #     # vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h #?not used 
        #     # vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]]  #1i #?not used 
        #     vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
        #     vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

        @inbounds begin 
                    #             # A .+= membuf[5]*vec_1h*transpose(vec_1h) #add auglag for 1h #?not used
                                
                    #             # A .+= membuf[5]*vec_1i*transpose(vec_1i) #add auglag for 1i #? not used 
                                
                    #             A .+= membuf[5]*vec_1j*transpose(vec_1j) #add auglag for 1j
                                
                    #             A .+= membuf[5]*vec_1k*transpose(vec_1k) #add auglag for 1k
                
                
                    for i = 3:8
                        for j = 3:8
                            A_aug[i,j] = Hbr[i-2,j-2]
                            # @cuprintln(A_aug[i,j])
                        end
                    end    
                    
                    # if lineidx == 2
                    #     @cuprintln("buf =", membuf[5,lineidx])
                    # end
                    
                    for i=1:8
                        for j=1:8
                            A_aug[i,j] += membuf[5,lineidx]*vec_1j[i]*vec_1j[j]
                            A_aug[i,j] += membuf[5,lineidx]*vec_1k[i]*vec_1k[j]  
                        end
                        # if lineidx == 2
                        #     @cuprintln(vec_1j[i])
                        #     @cuprintln(vec_1k[i])
                        # end
                    end

                    # if lineidx == 2
                    #     @cuprintln("second:")
                    #     for i = 1:8
                    #         for j = 1:8
                    #             @cuprintln(A_aug[i,j])
                    #         end
                    #     end
                    # end 
                
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
        #     return A, transpose(C)*A*C*scale #dim 8*8 no scale + dim = 6*6 scaled for red
    return
end

# function eval_b_auglag_branch_kernel_cpu_qpsub_red(
#     A_aug::Array{Float64,2}, bbr::Array{Float64,1}, l::Array{Float64,1}, rho::Array{Float64,1}, 
#     v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
#     YffR::Float64, YffI::Float64,
#     YftR::Float64, YftI::Float64,
#     YttR::Float64, YttI::Float64,
#     YtfR::Float64, YtfI::Float64, LH_1h::Array{Float64,1}, RH_1h::Float64,
#     LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64, scale::Float64)

#     b = zeros(8)
#     b[3:8] = bbr
    
    
#     #pij qij pji qji wrt branch structure ExaTron
#     supY = [0 0 YftR YftI YffR 0 0 0;
#     0 0 -YftI YftR -YffI 0 0 0;
#     0 0 YtfR -YtfI 0 YttR 0 0;
#     0 0 -YtfI -YtfR 0 -YttI 0 0]

#     #ALM on equality constraint wrt branch structure ExaTron
#     # vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h #not used 
#     # vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]] #1i #not used 
#     vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
#     vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

#     @inbounds begin 
#             # b .+= (membuf[1] - membuf[5]*RH_1h)*vec_1h #1h #?not used
#             # b .+= (membuf[2] - membuf[5]*RH_1i)*vec_1i #1i #?not used 
#             b .+= (membuf[3] - membuf[5]*RH_1j)*vec_1j #1j
#             b .+= (membuf[4] - membuf[5]*RH_1k)*vec_1k #1k
#     end

#     inv_ij = inv([LH_1h[1] LH_1h[2]; LH_1i[1] LH_1i[2]]) #TODO: fix with computation 
#     C_ij = -inv_ij * [0 0 LH_1h[3] LH_1h[4] 0 0; 0 0 0 0 LH_1i[3] LH_1i[4]]
#     d_ij = inv_ij * [RH_1h; RH_1i]
#     C = zeros(8,6)
#     C[1,1] = C[2,2] = 1
#     C[3:4,:] .= C_ij
#     C[5,3] = 1
#     C[6,4] = 1
#     C[7,5] = 1
#     C[8,6] = 1

#     d=zeros(8)
#     d[3:4] .= d_ij

      



#     return C, d, transpose(C) * (A_aug * d  + b)*scale #dims = 8*6, 8, 6, coeff, coeff, scaled 
# end