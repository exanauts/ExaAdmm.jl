"""
    auglag_linelimit_two_level_alternative_qpsub_ij()

- for certain line (i,j), update sol.u[pij_idx]
- use Exatron, eval_A_auglag_branch_kernel_cpu_qpsub, eval_b_auglag_branch_kernel_cpu_qpsub, build_QP_DS
- LANCELOT ALM algorithm 
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


function auglag_Ab_linelimit_two_level_alternative_qpsub_ij(
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    Hbr::Array{Float64,2}, bbr::Array{Float64,1}, lqp::Array{Float64,1}, uqp::Array{Float64,1},
    l::Array{Float64,1}, rho::Array{Float64,1}, 
    u::Array{Float64,1}, shift_idx::Int, v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,1},
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, 
    LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64)

    # variable wrt branch structure wrt Exatron
    x = zeros(8) #! initialization 
    f = 0.0
    xl = [0.0; 0.0; lqp]
    xu = [200000.0; 200000.0; uqp]


    # initialization on penalty
    if major_iter == 1 #info.inner = 1 (first inner iteration)
        membuf[5] = 10.0 #reset ρ_1h = ρ_1i = ρ_1j = ρ_1k (let ρ the same for all AL terms)
        mu = 10.0 # set mu = initial ρ_*      
    else
        mu = membuf[5] #inherit mu = ρ from the previous inner iteration
    end    

        
    # set internal parameters eta omega mu to guide ALM convergence
    eta = 1 / mu^0.1
    omega = 1 / mu
  
        
    it = 0 #iteration count
    terminate = false #termination status
    
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

        while !terminate
            it += 1
            
            #create QP parameters SCALED
            Atron = eval_A_auglag_branch_kernel_cpu_qpsub(Hbr,l, rho, v, z_curr, membuf,
            YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, 
            LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k,RH_1k,scale)
            btron =  eval_b_auglag_branch_kernel_cpu_qpsub(bbr,l, rho, v, z_curr, membuf,
            YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, 
            LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k,RH_1k,scale)
            tron = build_QP_DS(scale*Atron,scale*btron,xl,xu)                                     
            tron.x .= x #initialization on tron 

            # Solve the branch problem.
            status = ExaTron.solveProblem(tron)
            x .= tron.x
            f = tron.f #! wont match with IPOPT since constant terms are ignored
            fdot = 0.5*dot(x,Atron,x) + dot(btron,x)

            
            # violation on 1h,1i,1j,1k
            cviol1 = dot(vec_1h, x) - RH_1h 
            cviol2 = dot(vec_1i, x) 
            cviol3 = dot(vec_1j, x) - RH_1j
            cviol4 = dot(vec_1k, x) - RH_1k

            cnorm = max(abs(cviol1), abs(cviol2), abs(cviol3), abs(cviol4)) #worst violation
            
            #print current tron result of current iteration of ALM  
            println("it = ",it, " with cnorm = ",cnorm, " and solx = ",x)
            println("obj from tron =", f," obj from dot = ",fdot)
            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    membuf[1] += mu*cviol1 #λ_1h
                    membuf[2] += mu*cviol2 #λ_1i
                    membuf[3] += mu*cviol3 #λ_1j
                    membuf[4] += mu*cviol4 #λ_1k

                    eta = eta / mu^0.9
                    omega  = omega / mu
                end
            else
                mu = min(mu_max, mu*10) #increase penalty
                eta = 1 / mu^0.1
                omega = 1 / mu
                membuf[5] = mu #save penalty for current inner iteration 
            end

            if it >= max_auglag #maximum iteration for auglag 
                println("max_auglag reached for line with cnorm = ", cnorm)
                terminate = true
            end
        end #end ALM

    #save variables TODO: check if sol.u is actually updated 
    u[shift_idx] = dot(supY[1,:],x) #pij
    u[shift_idx + 1] = dot(supY[2,:],x) #qij
    u[shift_idx + 2] = dot(supY[3,:],x) #pji
    u[shift_idx + 3] = dot(supY[4,:],x) #qji
    u[shift_idx + 4] = x[5]
    u[shift_idx + 5] = x[6]
    u[shift_idx + 6] = x[7]
    u[shift_idx + 7] = x[8]

    
    # return solution and objective for branch testing     
    return x, f 
end
