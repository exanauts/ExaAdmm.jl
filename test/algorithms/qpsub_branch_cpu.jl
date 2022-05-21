# using Test
# using LazyArtifacts
# using LinearAlgebra
# using Printf
# # using CUDA

# using ExaAdmm
# using Random
# using JuMP
# using Ipopt

@testset "Testing branch kernel updates" begin

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

Random.seed!(10);
res = zeros(mod.grid_data.nline)

@inbounds begin 
    for i = 1: mod.grid_data.nline
        
        shift_idx = 6 + 8*(i-1)
        Hs = rand(6,6) 
        Hs = Hs * transpose(Hs) #make symmetric, inherit structure Hessian

        #inherit structure Linear Constraint: ignore 1h and 1i for testing 
        LH_1h = zeros(4) #LH * x = RH
        RH_1h = 0.0 
        Cf_1i = zeros(6) #Cf * x = 0
        
        #inherit structure line limit 
        LH_1j = rand(2)
        RH_1j = sum(rand(1)) + 100 #reduce dim
        LH_1k = rand(2)
        RH_1k = sum(rand(1)) + 100 #reduce dim

        #inherit structure bound
        ls = rand(6)
        us = ls .+ 10 

        supY = [mod.grid_data.YftR[i]  mod.grid_data.YftI[i]  mod.grid_data.YffR[i] 0 0 0;
        -mod.grid_data.YftI[i]  mod.grid_data.YftR[i]  -mod.grid_data.YffI[i] 0 0 0;
        mod.grid_data.YtfR[i]  -mod.grid_data.YtfI[i]  0  mod.grid_data.YttR[i] 0 0;
        -mod.grid_data.YtfI[i]  -mod.grid_data.YtfR[i] 0  -mod.grid_data.YttI[i] 0 0]

        A_ipopt = ExaAdmm.eval_A_branch_kernel_cpu_qpsub(Hs, sol.l_curr[shift_idx + 1: shift_idx + 8], 
        sol.rho[shift_idx + 1: shift_idx + 8], sol.v_curr[shift_idx + 1: shift_idx + 8], 
        sol.z_curr[shift_idx + 1: shift_idx + 8], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])

        b_ipopt = ExaAdmm.eval_b_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx + 1: shift_idx + 8], 
        sol.rho[shift_idx + 1: shift_idx + 8], sol.v_curr[shift_idx + 1: shift_idx + 8], 
        sol.z_curr[shift_idx + 1: shift_idx + 8], 
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i])

        l_ipopt = ls
        u_ipopt = us
        
        # call Ipopt
        model = JuMP.Model(Ipopt.Optimizer)
        # set_silent(model)
        @variable(model, l_ipopt[i]<= x[i=1:6] <=u_ipopt[i])
        @objective(model, Min, 0.5 * dot(x, A_ipopt, x) + dot(b_ipopt, x))
        # @constraint(model, eq1h, dot(LH_1h, x[1:4]) == RH_1h )
        @constraint(model, sij, LH_1j[1] * dot(supY[1,:],x) + LH_1j[2] * dot(supY[2,:],x) <= RH_1j)
        @constraint(model, sji,  LH_1k[1] * dot(supY[3,:],x) + LH_1k[2] * dot(supY[4,:],x) <= RH_1k)
        optimize!(model)
        # x_ipopt = value.(x)
        println("objective = ", objective_value(model), " with solution = ",value.(x))

        #test eval for auglag (tested)
        # A_auglag = ExaAdmm.eval_A_auglag_branch_kernel_cpu_qpsub(Hs, sol.l_curr[shift_idx + 1: shift_idx + 8], 
        # sol.rho[shift_idx + 1: shift_idx + 8], sol.v_curr[shift_idx + 1: shift_idx + 8], 
        # sol.z_curr[shift_idx + 1: shift_idx + 8], mod.qpsub_membuf[:,i],
        # mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        # mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        # mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        # mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],LH_1h,RH_1h,Cf_1i,LH_1j,RH_1j,LH_1k,RH_1k)

        # b_auglag = ExaAdmm.eval_b_auglag_branch_kernel_cpu_qpsub(sol.l_curr[shift_idx + 1: shift_idx + 8], 
        # sol.rho[shift_idx + 1: shift_idx + 8], sol.v_curr[shift_idx + 1: shift_idx + 8], 
        # sol.z_curr[shift_idx + 1: shift_idx + 8], mod.qpsub_membuf[:,i],
        # mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        # mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        # mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        # mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],LH_1h,RH_1h,Cf_1i,LH_1j,RH_1j,LH_1k,RH_1k)

        tronx, tronf = ExaAdmm.auglag_Ab_linelimit_two_level_alternative_qpsub_ij(par.max_auglag, par.mu_max, A_ipopt, b_ipopt, ls, us, sol.l_curr[shift_idx + 1: shift_idx + 8], 
        sol.rho[shift_idx + 1: shift_idx + 8], sol.v_curr[shift_idx + 1: shift_idx + 8], 
        sol.z_curr[shift_idx + 1: shift_idx + 8], mod.qpsub_membuf[:,i],
        mod.grid_data.YffR[i], mod.grid_data.YffI[i],
        mod.grid_data.YftR[i], mod.grid_data.YftI[i],
        mod.grid_data.YttR[i], mod.grid_data.YttI[i],
        mod.grid_data.YtfR[i], mod.grid_data.YtfI[i],LH_1h,RH_1h,Cf_1i,LH_1j,RH_1j,LH_1k,RH_1k)
        
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
   
end #testset