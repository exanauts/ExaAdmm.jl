"""
    eval_A_*(), eval_b_*()

- prepare call backs for build_QP_DS and IPOPT benchmark (solve branch kernel directly)
- use mod.membuf (see model.jl)
- TODO: membuf and better structure for memory
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















function eval_A_branch_kernel_cpu_qpsub(
    H::Array{Float64,2},l::Array{Float64,1}, rho::Array{Float64,1}, v::Array{Float64,1}, z_curr::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64)
    
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #linear transform pij qij pji qji wrt Hessian inherited structure 
    supY = [YftR YftI YffR 0 0 0;
    -YftI YftR -YffI 0 0 0;
    YtfR -YtfI 0 YttR 0 0;
    -YtfI -YtfR 0 -YttI 0 0]

    H_new = H
    @inbounds begin 
            # println(H)
            H_new .+= rho[1]*supY[1,:]*transpose(supY[1,:]) #pij
            H_new .+= rho[2]*supY[2,:]*transpose(supY[2,:]) #qij
            H_new .+= rho[3]*supY[3,:]*transpose(supY[3,:]) #pji
            H_new .+= rho[4]*supY[4,:]*transpose(supY[4,:]) #qji
            # println(H)
            H_new[3,3] += rho[5] #wi(ij) 
            H_new[4,4] += rho[6] #wj(ji) 
            H_new[5,5] += rho[7] #thetai(ij)
            H_new[6,6] += rho[8] #thetaj(ji)
    end
    
    #! H may not be perfectly symmetric 
    return H #6*6
end









function eval_A_auglag_branch_kernel_cpu_qpsub(
    Hbr::Array{Float64,2},l::Array{Float64,1}, rho::Array{Float64,1}, 
    v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, 
    LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64,scale::Float64)

    A = zeros(8,8)
    A[3:8,3:8] = Hbr
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #pij qij pji qji wrt branch structure ExaTron
    supY = [0 0 YftR YftI YffR 0 0 0;
    0 0 -YftI YftR -YffI 0 0 0;
    0 0 YtfR -YtfI 0 YttR 0 0;
    0 0 -YtfI -YtfR 0 -YttI 0 0]

    #ALM on equality constraint wrt branch structure ExaTron
    vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h
    vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]]  #1i
    vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
    vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

    @inbounds begin 
            A .+= membuf[5]*vec_1h*transpose(vec_1h) #add auglag for 1h
            
            A .+= membuf[5]*vec_1i*transpose(vec_1i) #add auglag for 1i
            
            A .+= membuf[5]*vec_1j*transpose(vec_1j) #add auglag for 1j
            
            A .+= membuf[5]*vec_1k*transpose(vec_1k) #add auglag for 1k
    end
    
    #! A may not be perfectly symmetric 
    return A*scale #dim = 8*8 
end



function eval_A_auglag_branch_kernel_cpu_qpsub_red(
    Hbr::Array{Float64,2},l::Array{Float64,1}, rho::Array{Float64,1}, 
    v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, 
    LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64,scale::Float64)

    A = zeros(8,8)
    A[3:8,3:8] = Hbr
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #pij qij pji qji wrt branch structure ExaTron
    supY = [0 0 YftR YftI YffR 0 0 0;
    0 0 -YftI YftR -YffI 0 0 0;
    0 0 YtfR -YtfI 0 YttR 0 0;
    0 0 -YtfI -YtfR 0 -YttI 0 0]

    #ALM on equality constraint wrt branch structure ExaTron
    # vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h #?not used 
    # vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]]  #1i #?not used 
    vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
    vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

    @inbounds begin 
            # A .+= membuf[5]*vec_1h*transpose(vec_1h) #add auglag for 1h #?not used
            
            # A .+= membuf[5]*vec_1i*transpose(vec_1i) #add auglag for 1i #? not used 
            
            A .+= membuf[5]*vec_1j*transpose(vec_1j) #add auglag for 1j
            
            A .+= membuf[5]*vec_1k*transpose(vec_1k) #add auglag for 1k
    end
    
    #! A may not be perfectly symmetric 

    inv_ij = inv([LH_1h[1]  LH_1h[2]; LH_1i[1]  LH_1i[2]]) #TODO: fix with computation 
    C_ij = -inv_ij * [0  0  LH_1h[3] LH_1h[4] 0 0; 0 0 0 0 LH_1i[3] LH_1i[4]]
    C = zeros(8,6)
    C[1,1] = C[2,2] = 1
    C[3:4,:] .= C_ij
    C[5,3] = 1
    C[6,4] = 1
    C[7,5] = 1
    C[8,6] = 1

    return A*scale, transpose(C)*A*C*scale #dim 8*8 and dim = 6*6 
end






function eval_b_branch_kernel_cpu_qpsub(
    l::Array{Float64,1}, rho::Array{Float64,1}, v::Array{Float64,1}, z::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64)
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    supY = [YftR YftI YffR 0 0 0;
    -YftI YftR -YffI 0 0 0;
    YtfR -YtfI 0 YttR 0 0;
    -YtfI -YtfR 0 -YttI 0 0]

    b = zeros(6)

    @inbounds begin
        b .+= (l[1] - rho[1]*(v[1]-z[1])) * supY[1,:] #pij
        b .+=  (l[2] - rho[2]*(v[2]-z[2])) * supY[2,:] #qij
        b .+= (l[3] - rho[3]*(v[3]-z[3])) * supY[3,:] #pji
        b .+=  (l[4] - rho[4]*(v[4]-z[4])) * supY[4,:] #qji
        b[3] += (l[5] - rho[5]*(v[5]-z[5])) #wi(ij)
        b[4] += (l[6] - rho[6]*(v[6]-z[6])) #wj(ji)
        b[5] += (l[7] - rho[7]*(v[7]-z[7])) #thetai(ij)
        b[6] += (l[8] - rho[8]*(v[8]-z[8])) #thetaj(ji)
    end

    return b #size 6
end














function eval_b_auglag_branch_kernel_cpu_qpsub(
    bbr::Array{Float64,1}, l::Array{Float64,1}, rho::Array{Float64,1}, 
    v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64, scale::Float64)

    b = zeros(8)
    b[3:8] = bbr
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #pij qij pji qji wrt branch structure ExaTron
    supY = [0 0 YftR YftI YffR 0 0 0;
    0 0 -YftI YftR -YffI 0 0 0;
    0 0 YtfR -YtfI 0 YttR 0 0;
    0 0 -YtfI -YtfR 0 -YttI 0 0]

    #ALM on equality constraint wrt branch structure ExaTron
    vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h
    vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]] #1i
    vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
    vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

    @inbounds begin 
            b .+= (membuf[1] - membuf[5]*RH_1h)*vec_1h #1h
            b .+= (membuf[2] - membuf[5]*RH_1i)*vec_1i #1i
            b .+= (membuf[3] - membuf[5]*RH_1j)*vec_1j #1j
            b .+= (membuf[4] - membuf[5]*RH_1k)*vec_1k #1k
    end

    return b*scale #dim = 8
end


function eval_b_auglag_branch_kernel_cpu_qpsub_red(
    A_aug::Array{Float64,2}, bbr::Array{Float64,1}, l::Array{Float64,1}, rho::Array{Float64,1}, 
    v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64, scale::Float64)

    b = zeros(8)
    b[3:8] = bbr
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #pij qij pji qji wrt branch structure ExaTron
    supY = [0 0 YftR YftI YffR 0 0 0;
    0 0 -YftI YftR -YffI 0 0 0;
    0 0 YtfR -YtfI 0 YttR 0 0;
    0 0 -YtfI -YtfR 0 -YttI 0 0]

    #ALM on equality constraint wrt branch structure ExaTron
    # vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h #not used 
    # vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]] #1i #not used 
    vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
    vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

    @inbounds begin 
            # b .+= (membuf[1] - membuf[5]*RH_1h)*vec_1h #1h #?not used
            # b .+= (membuf[2] - membuf[5]*RH_1i)*vec_1i #1i #?not used 
            b .+= (membuf[3] - membuf[5]*RH_1j)*vec_1j #1j
            b .+= (membuf[4] - membuf[5]*RH_1k)*vec_1k #1k
    end

    inv_ij = inv([LH_1h[1] LH_1h[2]; LH_1i[1] LH_1i[2]]) #TODO: fix with computation 
    C_ij = -inv_ij * [0 0 LH_1h[3] LH_1h[4] 0 0; 0 0 0 0 LH_1i[3] LH_1i[4]]
    d_ij = inv_ij * [RH_1h; RH_1i]
    C = zeros(8,6)
    C[1,1] = C[2,2] = 1
    C[3:4,:] .= C_ij
    C[5,3] = 1
    C[6,4] = 1
    C[7,5] = 1
    C[8,6] = 1

    d=zeros(8)
    d[3:4] .= d_ij

      



    return C, d, transpose(C) * (A_aug * d  + b)*scale #dims = 8*6, 8, 6
end