# @testset "Testing [x,xbar,z,l,lz] updates and a solve for ACOPF" begin
@testset "Testing [x,xbar,z,l,lz] updates" begin
    # Testing case9.m
    case = joinpath(INSTANCES_DIR, "case9.m")
    rho_pq = 4e2; rho_va = 4e4
    atol = 1e-6; verbose=0

    # Initialize an ACOPF model with default options.
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; verbose=verbose)
    mod = ExaAdmm.ModelQpsub{T,TD,TI,TM}(env)
    sol = mod.solution

    env.params.scale = 1e-4
    env.params.initial_beta = 1e3
    env.params.beta = 1e3

    # We'll perform just one ADMM iteration. Do some pre-steps first.
    ExaAdmm.admm_increment_outer(env, mod)
    ExaAdmm.admm_outer_prestep(env, mod)
    ExaAdmm.admm_increment_reset_inner(env, mod)
    ExaAdmm.admm_increment_inner(env, mod)
    ExaAdmm.admm_inner_prestep(env, mod)

    # x update
    ExaAdmm.admm_update_x(env, mod)

    # Referece solution. Entries are in the order of their indices.
    U_GEN = [0.1, -0.0, 0.2380952, -0.0, 0.1614035, -0.0]
    U_BR  = [0.0, 0.0, 0.0, 0.0, 1.01, 1.01, 0.0, 0.0, 0.0, 
    -0.079785, -0.0, -0.079785, 1.009937, 1.009937, 0.0, -0.0, 0.0, -0.1807321, 
    -0.0, -0.1807321, 1.0096765, 1.0096765, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 1.01, 1.01, 0.0, 
    0.0, 0.0, -0.1055335, -0.0, -0.1055335, 1.0098897, 1.0098897, 0.0, 
    -0.0, -0.0, -0.0752408, 0.0, -0.0752408, 1.009944, 1.009944, -0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.01, 1.01, 0.0, 0.0, -0.0, -0.1544938, 0.0, 
    -0.1544938, 1.0097636, 1.0097636, -0.0, 0.0, 0.0, -0.0888731, -0.0, -0.0888731, 1.0099218, 1.0099218, 0.0, -0.0]

    @test norm(sol.u_curr[1:2]   .- U_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(sol.u_curr[3:4]   .- U_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(sol.u_curr[5:6]   .- U_GEN[5:6],  Inf) <= atol # Gen 3
    # @test norm(sol.u_curr[7:14]  .- U_BR[1:8],   Inf) <= atol # Line 1
    # @test norm(sol.u_curr[15:22] .- U_BR[9:16],  Inf) <= atol # Line 2
    # @test norm(sol.u_curr[23:30] .- U_BR[17:24], Inf) <= atol # Line 3
    # @test norm(sol.u_curr[31:38] .- U_BR[25:32], Inf) <= atol # Line 4
    # @test norm(sol.u_curr[39:46] .- U_BR[33:40], Inf) <= atol # Line 5
    # @test norm(sol.u_curr[47:54] .- U_BR[41:48], Inf) <= atol # Line 6
    # @test norm(sol.u_curr[55:62] .- U_BR[49:56], Inf) <= atol # Line 7
    # @test norm(sol.u_curr[63:70] .- U_BR[57:64], Inf) <= atol # Line 8
    # @test norm(sol.u_curr[71:78] .- U_BR[65:72], Inf) <= atol # Line 9

    # xbar update
    ExaAdmm.admm_update_xbar(env, mod)

    V_GEN = [0.05, 0.0, 0.1190476, 0.0, 0.0807018, 0.0]
    V_BR  = [0.05, 0.0, -0.0, 0.0562194, 1.01, 1.0099529, 0.0, 0.0, 0.0, -0.0235656, -0.45, -0.0995265, 
    1.0099529, 1.0098067, 0.0, 0.0, -0.45, -0.2004735, -0.0, -0.0853102, 1.0098067, 1.0098554, 0.0, -0.0, 0.0807018, 
    0.0, 0.0, 0.0954219, 1.01, 1.0098554, 0.0, -0.0, 0.0, -0.0101116, -0.5, -0.1901463, 1.0098554, 1.0099168, -0.0,
     -0.0, -0.5, -0.1598537, 0.0, 0.0013374, 1.0099168, 1.0099025, -0.0, 0.0, -0.0, 0.0765782, 0.1190476, 0.0, 
     1.0099025, 1.01, 0.0, 0.0, -0.0, -0.0779156, -0.625, -0.2828104, 1.0099025, 1.0098427, 0.0, 0.0, -0.625, 
     -0.2171896, -0.0, -0.0326537, 1.0098427, 1.0099529, 0.0, 0.0]

    @test norm(sol.v_curr[1:2]   .- V_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(sol.v_curr[3:4]   .- V_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(sol.v_curr[5:6]   .- V_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(sol.v_curr[7:14]  .- V_BR[1:8],   Inf) <= atol # Line 1
    @test norm(sol.v_curr[15:22] .- V_BR[9:16],  Inf) <= atol # Line 2
    @test norm(sol.v_curr[23:30] .- V_BR[17:24], Inf) <= atol # Line 3
    @test norm(sol.v_curr[31:38] .- V_BR[25:32], Inf) <= atol # Line 4
    @test norm(sol.v_curr[39:46] .- V_BR[33:40], Inf) <= atol # Line 5
    @test norm(sol.v_curr[47:54] .- V_BR[41:48], Inf) <= atol # Line 6
    @test norm(sol.v_curr[55:62] .- V_BR[49:56], Inf) <= atol # Line 7
    @test norm(sol.v_curr[63:70] .- V_BR[57:64], Inf) <= atol # Line 8
    @test norm(sol.v_curr[71:78] .- V_BR[65:72], Inf) <= atol # Line 9

    # z update
    ExaAdmm.admm_update_z(env, mod)

    Z_GEN = [-0.0142857, -0.0, -0.0340136, -0.0, -0.0230576, -0.0]
    Z_BR  = [0.0142857, -0.0, -0.0, 0.0160627, -0.0, -4.59e-5, -0.0, 0.0, -0.0, 0.0160627, -0.1285714, 
    -0.0056404, 1.56e-5, -0.0001271, -0.0, 0.0, -0.1285714, -0.0056404, 0.0, 0.0272634, 0.0001271, 0.0001745, 
    -0.0, 0.0, 0.0230576, -0.0, 0.0, 0.0272634, -0.0, -0.0001411, -0.0, -0.0, 0.0, 0.0272634, -0.1428571,
     -0.0241751, -3.35e-5, 2.65e-5, -0.0, -0.0, -0.1428571, -0.0241751, -0.0, 0.0218795, -2.65e-5, -4.04e-5, 0.0,
      -0.0, -0.0, 0.0218795, 0.0340136, -0.0, -9.51e-5, -0.0, 0.0, -0.0, -0.0, 0.0218795, -0.1785714, -0.0366619, 
      0.0001355, 7.72e-5, 0.0, 0.0, -0.1785714, -0.0366619, -0.0, 0.0160627, -7.72e-5, 3.04e-5, -0.0, 0.0]

    @test norm(sol.z_curr[1:2]   .- Z_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(sol.z_curr[3:4]   .- Z_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(sol.z_curr[5:6]   .- Z_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(sol.z_curr[7:14]  .- Z_BR[1:8],   Inf) <= atol # Line 1
    @test norm(sol.z_curr[15:22] .- Z_BR[9:16],  Inf) <= atol # Line 2
    @test norm(sol.z_curr[23:30] .- Z_BR[17:24], Inf) <= atol # Line 3
    @test norm(sol.z_curr[31:38] .- Z_BR[25:32], Inf) <= atol # Line 4
    @test norm(sol.z_curr[39:46] .- Z_BR[33:40], Inf) <= atol # Line 5
    @test norm(sol.z_curr[47:54] .- Z_BR[41:48], Inf) <= atol # Line 6
    @test norm(sol.z_curr[55:62] .- Z_BR[49:56], Inf) <= atol # Line 7
    @test norm(sol.z_curr[63:70] .- Z_BR[57:64], Inf) <= atol # Line 8
    @test norm(sol.z_curr[71:78] .- Z_BR[65:72], Inf) <= atol # Line 9

    # l update
    ExaAdmm.admm_update_l(env, mod)

    L_GEN = [14.2857143, -0.0, 34.0136054, -0.0, 23.0576441, -0.0]
    L_BR  = [-14.2857143, -0.0, 0.0, -16.06268, -0.0, 0.0459313, -0.0, -0.0, 0.0, -16.06268, 
    128.5714286, 5.6404121, -0.0155616, 0.127064, 0.0, -0.0, 128.5714286, 5.6404121, -0.0, -27.2633876, -0.127064, 
    -0.1745497, 0.0, -0.0, -23.0576441, -0.0, -0.0, -27.2633876, -0.0, 0.1410712, -0.0, 0.0, -0.0, -27.2633876, 
    142.8571429, 24.1750998, 0.0334784, -0.0264528, 0.0, 0.0, 142.8571429, 24.1750998, 0.0, -21.8794915, 0.0264528,
     0.0404121, -0.0, 0.0, 0.0, -21.8794915, -34.0136054, -0.0, 0.0950994, -0.0, -0.0, -0.0, 0.0, -21.8794915, 
     178.5714286, 36.6618636, -0.1355115, -0.077155, -0.0, -0.0, 178.5714286, 36.6618636, 0.0, -16.06268, 0.077155, 
     -0.0303696, 0.0, -0.0]
    
    @test norm(sol.l_curr[1:2]   .- L_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(sol.l_curr[3:4]   .- L_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(sol.l_curr[5:6]   .- L_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(sol.l_curr[7:14]  .- L_BR[1:8],   Inf) <= atol # Line 1
    @test norm(sol.l_curr[15:22] .- L_BR[9:16],  Inf) <= atol # Line 2
    @test norm(sol.l_curr[23:30] .- L_BR[17:24], Inf) <= atol # Line 3
    @test norm(sol.l_curr[31:38] .- L_BR[25:32], Inf) <= atol # Line 4
    @test norm(sol.l_curr[39:46] .- L_BR[33:40], Inf) <= atol # Line 5
    @test norm(sol.l_curr[47:54] .- L_BR[41:48], Inf) <= atol # Line 6
    @test norm(sol.l_curr[55:62] .- L_BR[49:56], Inf) <= atol # Line 7
    @test norm(sol.l_curr[63:70] .- L_BR[57:64], Inf) <= atol # Line 8
    @test norm(sol.l_curr[71:78] .- L_BR[65:72], Inf) <= atol # Line 9

    # residual update
    ExaAdmm.admm_update_residual(env, mod)

    @test norm(sol.rp .- (sol.u_curr .- sol.v_curr .+ sol.z_curr), Inf) <= atol
    @test norm(sol.rd .- (sol.z_curr .- sol.z_prev),               Inf) <= atol
    @test norm(sol.Ax_plus_By .- (sol.u_curr .- sol.v_curr),       Inf) <= atol

    # lz update
    lz_prev = zeros(length(sol.lz))
    lz_prev .= sol.lz

    ExaAdmm.admm_update_lz(env, mod)

    @test norm(sol.lz .- (lz_prev .+ (env.params.beta .* sol.z_curr)), Inf) <= atol
    
    #full ADMM test
    # env, mod = ExaAdmm.solve_acopf(case; outer_iterlim=25, rho_pq=rho_pq, rho_va=rho_va, outer_eps=2*1e-5, verbose=verbose)
    # @test mod.info.status == :Solved
    # @test mod.info.outer == 20
    # @test mod.info.cumul == 705
    # @test isapprox(mod.info.objval, 5303.435; atol=1e-3)

    # case = joinpath(INSTANCES_DIR, "case118.m")
    # env, mod = ExaAdmm.solve_acopf(case; outer_iterlim=25, rho_pq=rho_pq, rho_va=rho_va, outer_eps=2*1e-5, verbose=0)
    # @test mod.info.status == :Solved
    # @test mod.info.outer == 20
    # @test mod.info.cumul == 1232
    # @test isapprox(mod.info.objval, 129645.676; atol=1e-3)
end
