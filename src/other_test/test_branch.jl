using Test
using LazyArtifacts
using LinearAlgebra
using Printf
# using CUDA

using ExaAdmm
using Random
using JuMP
using Ipopt

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

#save res for testing 
res = zeros(mod.grid_data.nline)

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
        LH_1j = rand(2)
        mod.LH_1j[i,:] .= LH_1j
        RH_1j = sum(rand(1)) + 100 #reduce dim to scalar
        mod.RH_1j[i] = RH_1j

        LH_1k = rand(2)
        mod.LH_1k[i,:] .= LH_1k
        RH_1k = sum(rand(1)) + 100 #reduce dim to scalar 
        mod.RH_1k[i] = RH_1k
        #inherit structure bound
        ls = rand(6)
        mod.ls[i,:] .= ls
        us = ls .+ 10 
        mod.us[i,:] .= us
    end
end #inbounds

@inbounds begin
    for i = 1: mod.grid_data.nline
        
        shift_idx = mod.line_start + 8*(i-1)

        supY = [mod.grid_data.YftR[i]  mod.grid_data.YftI[i]  mod.grid_data.YffR[i] 0 0 0;
        -mod.grid_data.YftI[i]  mod.grid_data.YftR[i]  -mod.grid_data.YffI[i] 0 0 0;
        mod.grid_data.YtfR[i]  -mod.grid_data.YtfI[i]  0  mod.grid_data.YttR[i] 0 0;
        -mod.grid_data.YtfI[i]  -mod.grid_data.YtfR[i] 0  -mod.grid_data.YttI[i] 0 0]

        A_ipopt = ExaAdmm.eval_A_branch_kernel_cpu_qpsub(mod.Hs[6*(i-1)+1:6*i,1:6], sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])

        b_ipopt = ExaAdmm.eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])

        l_ipopt = mod.ls[i,:]
        u_ipopt = mod.us[i,:]
        
        # call Ipopt
        model = JuMP.Model(Ipopt.Optimizer)
        # set_silent(model)
        @variable(model, l_ipopt[k]<= x[k=1:6] <=u_ipopt[k])
        @objective(model, Min, 0.5 * dot(x, A_ipopt, x) + dot(b_ipopt, x))
        # @constraint(model, eq1h, dot(LH_1h, x[1:4]) == RH_1h )
        @constraint(model, sij, mod.LH_1j[i,1] * dot(supY[1,:],x) + mod.LH_1j[i,2] * dot(supY[2,:],x) <= mod.RH_1j[i])
        @constraint(model, sji, mod.LH_1k[i,1] * dot(supY[3,:],x) + mod.LH_1k[i,2] * dot(supY[4,:],x) <= mod.RH_1k[i])
        optimize!(model)
        # x_ipopt = value.(x)
        println("objective = ", objective_value(model), " with solution = ",value.(x))

        println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_prev[pij]
        tronx, tronf = ExaAdmm.auglag_Ab_linelimit_two_level_alternative_qpsub_ij(1, par.max_auglag, par.mu_max, 1.0, A_ipopt, b_ipopt, mod.ls[i,:], mod.us[i,:], sol.l_curr[shift_idx : shift_idx + 7], 
        sol.rho[shift_idx : shift_idx + 7], sol.u_curr, shift_idx, sol.v_curr[shift_idx : shift_idx + 7], 
        sol.z_curr[shift_idx : shift_idx + 7], mod.qpsub_membuf[:,i],
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],
        mod.LH_1h[i,:], mod.RH_1h[i], mod.LH_1i[i,:], mod.RH_1i[i], mod.LH_1j[i,:], mod.RH_1j[i], mod.LH_1k[i,:], mod.RH_1k[i])
        println(sol.u_curr[shift_idx : shift_idx + 7]) #output u_curr[pij]

        res[i] = norm(tronx[3:8] - value.(x))

    end
end #inbounds

    @test res[1] <= atol # Line 1
    @test res[2] <= atol # Line 2
    @test res[3] <= atol # Line 3
    @test res[4] <= atol # Line 4
    @test res[5] <= atol # Line 5
    @test res[6] <= atol # Line 6
    @test res[7] <= atol # Line 7
    @test res[8] <= atol # Line 8
    @test res[9] <= atol # Line 9
   
    println(sol.u_curr)
end #testset