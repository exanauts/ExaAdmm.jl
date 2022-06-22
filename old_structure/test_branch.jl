using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

#save res for testing 

@testset "Testing branch kernel updates" begin

INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "mp_demand")

case = joinpath(INSTANCES_DIR, "case9.m")
rho_pq = 4e2; rho_va = 4e4
atol = 1e-6; verbose=0

# Initialize an qpsub model with default options.
T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
mod = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env)
sol = mod.solution
par = env.params

env.params.scale = 1e-4
env.params.initial_beta = 1e3
env.params.beta = 1e3

# We'll perform just one ADMM iteration. Do some pre-steps first.
ExaAdmm.admm_increment_outer(env, mod)
ExaAdmm.admm_outer_prestep(env, mod)
ExaAdmm.admm_increment_reset_inner(env, mod)
ExaAdmm.admm_increment_inner(env, mod)
ExaAdmm.admm_inner_prestep(env, mod)

res = zeros(mod.grid_data.nline)
res2 = zeros(mod.grid_data.nline)

@inbounds begin 
    #fix random seed and generate SQP param for mod
    Random.seed!(20);
    for i = 1: mod.grid_data.nline
        Hs = rand(6,6) 
        Hs = Hs * transpose(Hs) #make symmetric, inherit structure Hessian
        mod.Hs[6*(i-1)+1:6*i,1:6] .= Hs

        #inherit structure of Linear Constraint (overleaf): ignore 1h and 1i with zero assignment for testing 
        LH_1h = zeros(4) #LH * x = RH
        mod.LH_1h[i,:] .= LH_1h
        RH_1h = 0.0 
        mod.RH_1h[i] = RH_1h

        LH_1i = zeros(4) #Lf * x = RH
        mod.LH_1i[i,:] .= LH_1h
        RH_1i = 0.0
        mod.RH_1i[i] = RH_1i

        #inherit structure line limit 
        LH_1j = zeros(2) #rand(2)
        mod.LH_1j[i,:] .= LH_1j
        RH_1j = 0 #sum(rand(1)) + 100 #reduce dim to scalar
        mod.RH_1j[i] = RH_1j

        LH_1k = zeros(2) #rand(2)
        mod.LH_1k[i,:] .= LH_1k
        RH_1k = 0 #sum(rand(1)) + 100 #reduce dim to scalar 
        mod.RH_1k[i] = RH_1k
        #inherit structure bound
        ls = -1.5 #rand(6)
        mod.ls[i,:] .= ls
        us = 1.5 #ls .+ 10 
        mod.us[i,:] .= us
    end
    sol.rho .= 100
    sol.v_curr .= rand(mod.nvar)*10 .-20
    sol.z_curr .= 0
    sol.l_curr .= 0 #rand(mod.nvar)*10 .-20
end #inbounds

@inbounds begin
    for i = 1: 1 #mod.grid_data.nline
        
        shift_idx = mod.line_start + 8*(i-1)

        supY = [mod.grid_data.YftR[i]  mod.grid_data.YftI[i]  mod.grid_data.YffR[i] 0 0 0;
        -mod.grid_data.YftI[i]  mod.grid_data.YftR[i]  -mod.grid_data.YffI[i] 0 0 0;
        mod.grid_data.YtfR[i]  -mod.grid_data.YtfI[i]  0  mod.grid_data.YttR[i] 0 0;
        -mod.grid_data.YtfI[i]  -mod.grid_data.YtfR[i] 0  -mod.grid_data.YttI[i] 0 0]
        
        # print("1: ",mod.Hs[6*(i-1)+1:6*i,1:6])
        A_ipopt = ExaAdmm.eval_A_branch_kernel_cpu_qpsub(mod.Hs[6*(i-1)+1:6*i,1:6], sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])
        # print("2: ",mod.Hs[6*(i-1)+1:6*i,1:6])

        b_ipopt = ExaAdmm.eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])
        # println()
        # print("3 ",mod.Hs[6*(i-1)+1:6*i,1:6])

        l_ipopt = mod.ls[i,:]
        u_ipopt = mod.us[i,:]
        
        # call Ipopt check QP
        model = JuMP.Model(Ipopt.Optimizer)
        set_silent(model)
        @variable(model, l_ipopt[k]<= x[k=1:6] <=u_ipopt[k])
        @objective(model, Min, 0.5 * dot(x, A_ipopt, x) + dot(b_ipopt, x))
        @constraint(model, eq1h, dot(LH_1h, x[1:4]) == RH_1h )
        @constraint(model, sij, mod.LH_1j[i,1] * dot(supY[1,:],x) + mod.LH_1j[i,2] * dot(supY[2,:],x) <= mod.RH_1j[i])
        @constraint(model, sji, mod.LH_1k[i,1] * dot(supY[3,:],x) + mod.LH_1k[i,2] * dot(supY[4,:],x) <= mod.RH_1k[i])
        optimize!(model)
        x_ipopt1 = value.(x)
        println("objective = ", objective_value(model), " with solution = ",x_ipopt1)

        # call Ipopt check ALM
        model2 = JuMP.Model(Ipopt.Optimizer)
        set_silent(model2)
        @variable(model2, l_ipopt[k]<= x[k=1:6] <=u_ipopt[k])
        @objective(model2, Min, 0.5*dot(x,mod.Hs[6*(i-1)+1:6*i,1:6],x) + 
                    sol.l_curr[shift_idx]*dot(supY[1,:],x) + sol.l_curr[shift_idx + 1]*dot(supY[2,:],x) +
                    sol.l_curr[shift_idx + 2]*dot(supY[3,:],x) + sol.l_curr[shift_idx + 3]*dot(supY[4,:],x) +
                    sol.l_curr[shift_idx + 4]*x[3] + sol.l_curr[shift_idx + 5]*x[4] + sol.l_curr[shift_idx + 6]*x[5] + sol.l_curr[shift_idx + 7]*x[6] +
                    0.5*sol.rho[shift_idx]*(dot(supY[1,:],x) - sol.v_curr[shift_idx] + sol.z_curr[shift_idx])^2 +
                    0.5*sol.rho[shift_idx + 1]*(dot(supY[2,:],x) - sol.v_curr[shift_idx + 1] + sol.z_curr[shift_idx + 1])^2 +
                    0.5*sol.rho[shift_idx + 2]*(dot(supY[3,:],x) - sol.v_curr[shift_idx + 2] + sol.z_curr[shift_idx + 2])^2 +
                    0.5*sol.rho[shift_idx + 3]*(dot(supY[4,:],x) - sol.v_curr[shift_idx + 3] + sol.z_curr[shift_idx + 3])^2 +
                    0.5*sol.rho[shift_idx + 4]*(x[3]-sol.v_curr[shift_idx + 4] + sol.z_curr[shift_idx + 4])^2 + 
                    0.5*sol.rho[shift_idx + 5]*(x[4]-sol.v_curr[shift_idx + 5] + sol.z_curr[shift_idx + 5])^2 +
                    0.5*sol.rho[shift_idx + 6]*(x[5]-sol.v_curr[shift_idx + 6] + sol.z_curr[shift_idx + 6])^2 +
                    0.5*sol.rho[shift_idx + 7]*(x[6]-sol.v_curr[shift_idx + 7] + sol.z_curr[shift_idx + 7])^2 )
        optimize!(model2)
        x_ipopt2 = value.(x)
        println("objective = ", objective_value(model2), " with solution = ",x_ipopt2)

        # println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_prev[pij]
        tronx, tronf = ExaAdmm.auglag_Ab_linelimit_two_level_alternative_qpsub_ij(1, par.max_auglag, par.mu_max, 1.0, A_ipopt, b_ipopt, mod.ls[i,:], mod.us[i,:], mod.sqp_line, sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], mod.qpsub_membuf,i,
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],
        mod.LH_1h[i,:], mod.RH_1h[i], mod.LH_1i[i,:], mod.RH_1i[i], mod.LH_1j[i,:], mod.RH_1j[i], mod.LH_1k[i,:], mod.RH_1k[i])
        # println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_curr[pij]

        res[i] = norm(tronx[3:8] - x_ipopt1)
        res2[i] = norm(tronx[3:8] - x_ipopt2)

    end

end #inbounds

    # @test res[1] <= atol # Line 1
    # @test res[2] <= atol # Line 2
    # @test res[3] <= atol # Line 3
    # @test res[4] <= atol # Line 4
    # @test res[5] <= atol # Line 5
    # @test res[6] <= atol # Line 6
    # @test res[7] <= atol # Line 7
    # @test res[8] <= atol # Line 8
    # @test res[9] <= atol # Line 9
   
    println([res,res2])
end #testset