"""
    auglag_linelimit_two_level_alternative_qpsub_ij_red()

- for certain line (i,j), update sol.u[pij_idx]
- use Exatron, eval_A_auglag_branch_kernel_cpu_qpsub, eval_b_auglag_branch_kernel_cpu_qpsub, build_QP_DS
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


function auglag_Ab_linelimit_two_level_alternative_qpsub_ij_red(
    major_iter::Int, max_auglag::Int, mu_max::Float64, scale::Float64,
    Hbr::Array{Float64,2}, bbr::Array{Float64,1}, lqp::Array{Float64,1}, uqp::Array{Float64,1},sqp_line::Array{Float64,2},
    l::Array{Float64,1}, rho::Array{Float64,1}, 
    u::Array{Float64,1}, shift_idx::Int, v::Array{Float64,1}, z_curr::Array{Float64,1}, membuf::Array{Float64,2},lineidx::Int,
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64, 
    LH_1h::Array{Float64,1}, RH_1h::Float64,
    LH_1i::Array{Float64,1}, RH_1i::Float64, LH_1j::Array{Float64,1},RH_1j::Float64, LH_1k::Array{Float64,1},RH_1k::Float64, lambda::Array{Float64,2})

    
    #? for debug only
    # max_auglag = 4
    # cnorm_all = zeros(max_auglag)
    # eta_all = zeros(max_auglag)
    # mu_all = zeros(max_auglag)

    # variable wrt branch structure wrt Exatron
    x = [0.0; 0.0; sqp_line[3:6,lineidx]] #initialization 
    f = 0.0
    xl = [0.0; 0.0; lqp[3:6]]
    xu = [200000.0; 200000.0; uqp[3:6]]
    trg = zeros(6) #hold multiplier 
    
    Ctron = zeros(8,6)
    dtron = zeros(8)

    #? for debug
    # if lineidx == 1
    #     println("Hbr = ", Hbr)
    #     println("bbr = ", bbr)
    # end


    # initialization on penalty
    if major_iter == 1 #info.inner = 1 (first inner iteration)
        membuf[5,lineidx] = 10.0 #reset ρ_1h = ρ_1i = ρ_1j = ρ_1k (let ρ the same for all AL terms)
        mu = 10.0 # set mu = initial ρ_*      
    else
        mu = membuf[5,lineidx] #inherit mu = ρ from the previous inner iteration
    end    

        
    # set internal parameters eta omega mu to guide ALM convergence
    eta = 1 / mu^0.1
    # omega = 1 / mu #? not used 
  
        
    it = 0 #iteration count
    terminate = false #termination status
    
    # TODO: find better way to structure and speed up computation (e.g., membuf)
    #pij qij pji qji wrt branch structure ExaTron
    supY = [0 0 YftR YftI YffR 0 0 0;
        0 0 -YftI YftR -YffI 0 0 0;
        0 0 YtfR -YtfI 0 YttR 0 0;
        0 0 -YtfI -YtfR 0 -YttI 0 0]

    #ALM on equality constraint wrt branch structure ExaTron
    # vec_1h = [0, 0, LH_1h[1], LH_1h[2], LH_1h[3], LH_1h[4], 0, 0 ] #1h #? notused
    # vec_1i = [0, 0, LH_1i[1], LH_1i[2], 0, 0, LH_1i[3], LH_1i[4]] #1i #? notused 
    vec_1j = [1, 0, 0, 0, 0, 0, 0, 0] + LH_1j[1]* supY[1,:] + LH_1j[2]* supY[2,:] #1j with t_ij
    vec_1k = [0, 1, 0, 0, 0, 0, 0, 0] + LH_1k[1]* supY[3,:] + LH_1k[2]* supY[4,:] #1k with t_ji

    # println("vec1j = ", vec_1j)
    # println("vec1k = ", vec_1k)

        while !terminate
            it += 1
            
            #create QP parameters SCALED
            Atron, Atron_red = eval_A_auglag_branch_kernel_cpu_qpsub_red(Hbr,l, rho, v, z_curr, membuf[:,lineidx],
            YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, 
            LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k, RH_1k, scale)
            Ctron, dtron, btron_red =  eval_b_auglag_branch_kernel_cpu_qpsub_red(Atron, bbr,l, rho, v, z_curr, membuf[:,lineidx],
            YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, 
            LH_1h, RH_1h, LH_1i, RH_1i, LH_1j, RH_1j, LH_1k, RH_1k, scale)

            # if lineidx == 2 && it == 1
            #     println("Ctron ", Ctron)
            #     println("dtron ", dtron)
            #     println("A_aug ",Atron)
            #     println("Atron ",Atron_red)
            #     println("btron ",btron_red)
            # end



            # for debug only 
            # print(Atron)
            # print(btron)
            # print(xl)
            # print(xu)
            # print(membuf)

            tron = build_QP_DS(Atron_red,btron_red,xl,xu)                                     
            tron.x .= x #initialization on tron 

            # Solve the branch problem.
            status = ExaTron.solveProblem(tron)
            x .= tron.x
            sqp_line[:,lineidx] .= (Ctron * x + dtron)[3:8] #write to sqp_line
            f = tron.f #! wont match with IPOPT since constant terms are ignored
            trg = tron.g
            # fdot = 0.5*dot(x,Atron,x) + dot(btron,x) #? not used 
            
            
            # violation on 1h,1i,1j,1k
            # cviol1 = dot(vec_1h, x) - RH_1h #?not used
            # cviol2 = dot(vec_1i, x) - RH_1i #?not_used
            cviol3 = dot(vec_1j, Ctron * x + dtron) - RH_1j
            cviol4 = dot(vec_1k, Ctron * x + dtron) - RH_1k

            cnorm = max(abs(cviol3), abs(cviol4)) #worst violation
            
            #?debug only 
            # print current tron result of current iteration of ALM  
            # println("it = ",it, " with cnorm = ",[cviol1,cviol2,cviol3,cviol4], " and solx = ",x, "param =",[norm(Atron),norm(btron),xl,xu])
            # println("it = ",it, ", cnorm = ",cnorm, ", eta = ",eta, ", mu = ", mu )
            # println("obj from tron =", f," obj from dot = ",fdot)
            # cnorm_all[it] = cnorm
            # eta_all[it] = eta
            # mu_all[it] = mu

            if cnorm <= eta
                if cnorm <= 1e-6
                    terminate = true
                else
                    # membuf[1,lineidx] += mu*cviol1 #λ_1h #?not used 
                    # membuf[2,lineidx] += mu*cviol2 #λ_1i #?not used 
                    membuf[3,lineidx] += mu*cviol3 #λ_1j #?not reset (following YD)
                    membuf[4,lineidx] += mu*cviol4 #λ_1k #?not reset (following YD)

                    eta = eta / mu^0.9
                    # omega  = omega / mu #? not used 
                end
            else
                mu = min(mu_max, mu*10) #increase penalty
                # mu = min(mu_max, mu*(1/minimum([0.1,1/sqrt(mu)]))) #increase penalty from slides Gould, only increasing  
                eta = 1 / mu^0.1
                # omega = 1 / mu #? not used 
                membuf[5,lineidx] = mu #save penalty for current inner iteration 
            end

            if it >= max_auglag && cnorm > 1e-6 #maximum iteration for auglag 
                println()
                println("max_auglag reached for line with cnorm = ", cnorm, " for line = ", lineidx, " with mu ", mu)
                
                
                terminate = true
            end
            #?for debug
            # println(cnorm_all)
            # println()
            # println(eta_all)
            # println()
            # println(mu_all)
        end #end while ALM

       

    #save variables TODO: check if sol.u is actually updated 
    u[shift_idx] = dot(supY[1,:],Ctron * x + dtron) #pij
    u[shift_idx + 1] = dot(supY[2,:],Ctron * x + dtron) #qij
    u[shift_idx + 2] = dot(supY[3,:],Ctron * x + dtron) #pji
    u[shift_idx + 3] = dot(supY[4,:],Ctron * x + dtron) #qji
    u[shift_idx + 4] = x[3] #wi
    u[shift_idx + 5] = x[4] #wj
    u[shift_idx + 6] = x[5] #thetai
    u[shift_idx + 7] = x[6] #thetaj

    # supY = [0 0 YftR YftI YffR 0 0 0;
    #     0 0 -YftI YftR -YffI 0 0 0;
    #     0 0 YtfR -YtfI 0 YttR 0 0;
    #     0 0 -YtfI -YtfR 0 -YttI 0 0]
    #get multiplier
    tmpH = inv([LH_1h[1]  LH_1i[1]; LH_1h[2]  LH_1i[2]])
    tmp14_i = [2*u[shift_idx]*YftR + 2*u[shift_idx + 1]*(-YftI), 2*u[shift_idx]*YftI + 2*u[shift_idx + 1]*YftR]
    tmp14_h = [2*u[shift_idx + 2]*YtfR + 2*u[shift_idx + 3]*(-YtfI), 2*u[shift_idx + 2]*-YtfI + 2*u[shift_idx + 3]*(-YtfR)]
    #14h 14i
    lambda[1:2,lineidx] = -tmpH*(trg[1]*tmp14_i + trg[2]*tmp14_h + Hbr[1:2,1:2]*sqp_line[1:2,lineidx] + Hbr[1:2,3:6]*sqp_line[3:6,lineidx] + bbr[1:2]) 
    #14j 
    lambda[3,lineidx] = -abs(trg[1]) #<=0 one_side ineq
    #14k 
    lambda[4,lineidx] = -abs(trg[2]) #<=0 one-side ineq

    #for debug
    # println(tron.g) 

    
    # return solution and objective for branch testing     
    return Ctron * x + dtron, f 
end