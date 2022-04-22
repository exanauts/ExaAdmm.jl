@testset "Testing [x,xbar,z,l,lz] updates and a solve for MPACOPF" begin
    # Testing case9.m
    case = joinpath(INSTANCES_DIR, "case9.m")
    load_prefix = joinpath(MP_DEMAND_DIR, "case9_onehour_60")
    rho_pq = 4e2; rho_va = 4e4
    atol = 1e-6; verbose=0
    end_period = 3

    # Initialize a multi-period ACOPF model with time horizon of length 3.
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; load_prefix=load_prefix, verbose=verbose)
    mod = ExaAdmm.ModelMpacopf{T,TD,TI,TM}(env; end_period=end_period)
    sol = mod.solution

    # We'll perform just one ADMM iteration. Do some pre-steps first.
    ExaAdmm.admm_increment_outer(env, mod)
    ExaAdmm.admm_outer_prestep(env, mod)
    ExaAdmm.admm_increment_reset_inner(env, mod)
    ExaAdmm.admm_increment_inner(env, mod)
    ExaAdmm.admm_inner_prestep(env, mod)

    # x update
    ExaAdmm.admm_update_x(env, mod)

    U_GEN = [
        [
            0.100000, 0.0,
            0.238095, 0.0,
            0.161404, 0.0
        ],
        [
            0.173333, 0.0,
            0.438400, 0.0,
            0.307200, 0.0
        ],
        [
            0.173333, 0.0,
            0.438400, 0.0,
            0.307200, 0.0
        ]
    ]
    U_BR = [
        [
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.079790, 0.000000, -0.079790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.180790, 0.000000, -0.180790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.105545, 0.000000, -0.105545, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.075245, 0.000000, -0.075245, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.154530, 0.000000, -0.154530, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.088880, 0.000000, -0.088880, 1.010000, 1.010000, 0.000000, 0.000000
        ],
        [
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.079790, 0.000000, -0.079790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.180790, 0.000000, -0.180790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.105545, 0.000000, -0.105545, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.075245, 0.000000, -0.075245, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.154530, 0.000000, -0.154530, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.088880, 0.000000, -0.088880, 1.010000, 1.010000, 0.000000, 0.000000
        ],
        [
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.079790, 0.000000, -0.079790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.180790, 0.000000, -0.180790, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.105545, 0.000000, -0.105545, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.075245, 0.000000, -0.075245, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000,  0.000000, 0.000000,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.154530, 0.000000, -0.154530, 1.010000, 1.010000, 0.000000, 0.000000,
            0.000000, -0.088880, 0.000000, -0.088880, 1.010000, 1.010000, 0.000000, 0.000000
        ]
    ]
    @test norm(mod.models[1].solution.u_curr[1:2]   .- U_GEN[1][1:2],  Inf) <= atol # Gen 1 t=1
    @test norm(mod.models[1].solution.u_curr[3:4]   .- U_GEN[1][3:4],  Inf) <= atol # Gen 2 t=1
    @test norm(mod.models[1].solution.u_curr[5:6]   .- U_GEN[1][5:6],  Inf) <= atol # Gen 3 t=1
    @test norm(mod.models[2].solution.u_curr[1:2]   .- U_GEN[2][1:2],  Inf) <= atol # Gen 1 t=2
    @test norm(mod.models[2].solution.u_curr[3:4]   .- U_GEN[2][3:4],  Inf) <= atol # Gen 2 t=2
    @test norm(mod.models[2].solution.u_curr[5:6]   .- U_GEN[2][5:6],  Inf) <= atol # Gen 3 t=2
    @test norm(mod.models[3].solution.u_curr[1:2]   .- U_GEN[3][1:2],  Inf) <= atol # Gen 1 t=3
    @test norm(mod.models[3].solution.u_curr[3:4]   .- U_GEN[3][3:4],  Inf) <= atol # Gen 2 t=3
    @test norm(mod.models[3].solution.u_curr[5:6]   .- U_GEN[3][5:6],  Inf) <= atol # Gen 3 t=3
    @test norm(mod.models[1].solution.u_curr[7:14]  .- U_BR[1][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[1].solution.u_curr[15:22] .- U_BR[1][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[1].solution.u_curr[23:30] .- U_BR[1][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[1].solution.u_curr[31:38] .- U_BR[1][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[1].solution.u_curr[39:46] .- U_BR[1][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[1].solution.u_curr[47:54] .- U_BR[1][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[1].solution.u_curr[55:62] .- U_BR[1][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[1].solution.u_curr[63:70] .- U_BR[1][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[1].solution.u_curr[71:78] .- U_BR[1][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[2].solution.u_curr[7:14]  .- U_BR[2][1:8],   Inf) <= atol # Line 1 t=2
    @test norm(mod.models[2].solution.u_curr[15:22] .- U_BR[2][9:16],  Inf) <= atol # Line 2 t=2
    @test norm(mod.models[2].solution.u_curr[23:30] .- U_BR[2][17:24], Inf) <= atol # Line 3 t=2
    @test norm(mod.models[2].solution.u_curr[31:38] .- U_BR[2][25:32], Inf) <= atol # Line 4 t=2
    @test norm(mod.models[2].solution.u_curr[39:46] .- U_BR[2][33:40], Inf) <= atol # Line 5 t=2
    @test norm(mod.models[2].solution.u_curr[47:54] .- U_BR[2][41:48], Inf) <= atol # Line 6 t=2
    @test norm(mod.models[2].solution.u_curr[55:62] .- U_BR[2][49:56], Inf) <= atol # Line 7 t=2
    @test norm(mod.models[2].solution.u_curr[63:70] .- U_BR[2][57:64], Inf) <= atol # Line 8 t=2
    @test norm(mod.models[2].solution.u_curr[71:78] .- U_BR[2][65:72], Inf) <= atol # Line 9 t=2
    @test norm(mod.models[3].solution.u_curr[7:14]  .- U_BR[3][1:8],   Inf) <= atol # Line 1 t=3
    @test norm(mod.models[3].solution.u_curr[15:22] .- U_BR[3][9:16],  Inf) <= atol # Line 2 t=3
    @test norm(mod.models[3].solution.u_curr[23:30] .- U_BR[3][17:24], Inf) <= atol # Line 3 t=3
    @test norm(mod.models[3].solution.u_curr[31:38] .- U_BR[3][25:32], Inf) <= atol # Line 4 t=3
    @test norm(mod.models[3].solution.u_curr[39:46] .- U_BR[3][33:40], Inf) <= atol # Line 5 t=3
    @test norm(mod.models[3].solution.u_curr[47:54] .- U_BR[3][41:48], Inf) <= atol # Line 6 t=3
    @test norm(mod.models[3].solution.u_curr[55:62] .- U_BR[3][49:56], Inf) <= atol # Line 7 t=3
    @test norm(mod.models[3].solution.u_curr[63:70] .- U_BR[3][57:64], Inf) <= atol # Line 8 t=3
    @test norm(mod.models[3].solution.u_curr[71:78] .- U_BR[3][65:72], Inf) <= atol # Line 9 t=3

    # xbar update
    ExaAdmm.admm_update_xbar(env, mod)

    V_GEN = [
        [
            0.107778, 0.0,
            0.245498, 0.0,
            0.174201, 0.0
        ],
        [
            0.132222, 0.0,
            0.312267, 0.0,
            0.222800, 0.0
        ],
        [
            0.086667, 0.0,
            0.219200, 0.0,
            0.153600, 0.0
        ]
    ]
    V_BR = [
        [
             0.107777,  0.000000,  0.000000,  0.056223, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.023566, -0.450000, -0.099499, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.450000, -0.200500,  0.000000, -0.085345, 1.010000, 1.010000, 0.000000, 0.000000,
             0.174201,  0.000000,  0.000000,  0.095445, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.010100, -0.500000, -0.190150, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.500000, -0.159849,  0.000000,  0.001346, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000,  0.076591,  0.245498,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.077938, -0.625000, -0.282825, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.625000, -0.217174,  0.000000, -0.032656, 1.010000, 1.010000, 0.000000, 0.000000
        ],
        [
             0.132222,  0.000000,  0.000000,  0.056223, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.023566, -0.449824, -0.099441, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.449824, -0.200441,  0.000000, -0.085345, 1.010000, 1.010000, 0.000000, 0.000000,
             0.222800,  0.000000,  0.000000,  0.095445, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.010100, -0.499805, -0.190081, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.499805, -0.159781,  0.000000,  0.001346, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000,  0.076591,  0.312266,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.077938, -0.624756, -0.282727, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.624756, -0.217077,  0.000000, -0.032656, 1.010000, 1.010000, 0.000000, 0.000000,
        ],
        [
             0.086666,  0.000000,  0.000000,  0.056223, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.023566, -0.449686, -0.099395, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.449686, -0.200395,  0.000000, -0.085345, 1.010000, 1.010000, 0.000000, 0.000000,
             0.153599,  0.000000,  0.000000,  0.095445, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.010100, -0.499651, -0.190028, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.499651, -0.159728,  0.000000,  0.001346, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000,  0.076591,  0.219199,  0.000000, 1.010000, 1.010000, 0.000000, 0.000000,
             0.000000, -0.077938, -0.624564, -0.282650, 1.010000, 1.010000, 0.000000, 0.000000,
            -0.624564, -0.217000,  0.000000, -0.032656, 1.010000, 1.010000, 0.000000, 0.000000
        ]
    ]

    @test norm(mod.models[1].solution.v_curr[1:2]   .- V_GEN[1][1:2],  Inf) <= atol # Gen 1 t=1
    @test norm(mod.models[1].solution.v_curr[3:4]   .- V_GEN[1][3:4],  Inf) <= atol # Gen 2 t=1
    @test norm(mod.models[1].solution.v_curr[5:6]   .- V_GEN[1][5:6],  Inf) <= atol # Gen 3 t=1
    @test norm(mod.models[2].solution.v_curr[1:2]   .- V_GEN[2][1:2],  Inf) <= atol # Gen 1 t=2
    @test norm(mod.models[2].solution.v_curr[3:4]   .- V_GEN[2][3:4],  Inf) <= atol # Gen 2 t=2
    @test norm(mod.models[2].solution.v_curr[5:6]   .- V_GEN[2][5:6],  Inf) <= atol # Gen 3 t=2
    @test norm(mod.models[3].solution.v_curr[1:2]   .- V_GEN[3][1:2],  Inf) <= atol # Gen 1 t=3
    @test norm(mod.models[3].solution.v_curr[3:4]   .- V_GEN[3][3:4],  Inf) <= atol # Gen 2 t=3
    @test norm(mod.models[3].solution.v_curr[5:6]   .- V_GEN[3][5:6],  Inf) <= atol # Gen 3 t=3
    @test norm(mod.models[1].solution.v_curr[7:14]  .- V_BR[1][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[1].solution.v_curr[15:22] .- V_BR[1][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[1].solution.v_curr[23:30] .- V_BR[1][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[1].solution.v_curr[31:38] .- V_BR[1][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[1].solution.v_curr[39:46] .- V_BR[1][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[1].solution.v_curr[47:54] .- V_BR[1][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[1].solution.v_curr[55:62] .- V_BR[1][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[1].solution.v_curr[63:70] .- V_BR[1][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[1].solution.v_curr[71:78] .- V_BR[1][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[2].solution.v_curr[7:14]  .- V_BR[2][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[2].solution.v_curr[15:22] .- V_BR[2][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[2].solution.v_curr[23:30] .- V_BR[2][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[2].solution.v_curr[31:38] .- V_BR[2][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[2].solution.v_curr[39:46] .- V_BR[2][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[2].solution.v_curr[47:54] .- V_BR[2][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[2].solution.v_curr[55:62] .- V_BR[2][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[2].solution.v_curr[63:70] .- V_BR[2][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[2].solution.v_curr[71:78] .- V_BR[2][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[3].solution.v_curr[7:14]  .- V_BR[3][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[3].solution.v_curr[15:22] .- V_BR[3][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[3].solution.v_curr[23:30] .- V_BR[3][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[3].solution.v_curr[31:38] .- V_BR[3][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[3].solution.v_curr[39:46] .- V_BR[3][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[3].solution.v_curr[47:54] .- V_BR[3][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[3].solution.v_curr[55:62] .- V_BR[3][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[3].solution.v_curr[63:70] .- V_BR[3][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[3].solution.v_curr[71:78] .- V_BR[3][65:72], Inf) <= atol # Line 9 t=1

    # z update
    ExaAdmm.admm_update_z(env, mod)

    Z_GEN = [
        [
            0.002222, -0.000000,
            0.002115, -0.000000,
            0.003656, -0.000000
        ],
        [
            -0.011746, -0.000000,
            -0.036038, -0.000000,
            -0.024114, -0.000000
        ],
        [
            -0.024762, -0.000000,
            -0.062629, -0.000000,
            -0.043886, -0.000000
        ]
    ]
    Z_BR = [
        [
             0.030794, -0.000000,  0.000000,  0.016064, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.016064, -0.128571, -0.005631,  0.000000, -0.000000,  0.000000,  0.000000,
            -0.128571, -0.005631,  0.000000,  0.027270,  0.000000,  0.000000, -0.000000,  0.000000,
             0.049772, -0.000000,  0.000000,  0.027270, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.027270, -0.142857, -0.024173, -0.000000,  0.000000, -0.000000,  0.000000,
            -0.142857, -0.024173,  0.000000,  0.021883, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.021883,  0.070142, -0.000000, -0.000000, -0.000000,  0.000000, -0.000000,
             0.000000,  0.021883, -0.178571, -0.036656,  0.000000,  0.000000, -0.000000, -0.000000,
            -0.178571, -0.036656,  0.000000,  0.016064, -0.000000,  0.000000,  0.000000,  0.000000
        ],
        [
             0.037778, -0.000000,  0.000000,  0.016064, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.016064, -0.128521, -0.005615,  0.000000, -0.000000,  0.000000,  0.000000,
            -0.128521, -0.005615,  0.000000,  0.027270,  0.000000,  0.000000, -0.000000,  0.000000,
             0.063657, -0.000000,  0.000000,  0.027270, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.027270, -0.142801, -0.024153, -0.000000,  0.000000, -0.000000,  0.000000,
            -0.142801, -0.024153,  0.000000,  0.021883, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.021883,  0.089219, -0.000000, -0.000000, -0.000000,  0.000000, -0.000000,
             0.000000,  0.021883, -0.178502, -0.036628,  0.000000,  0.000000, -0.000000, -0.000000,
            -0.178502, -0.036628,  0.000000,  0.016064, -0.000000,  0.000000,  0.000000,  0.000000
        ],
        [
             0.024762, -0.000000,  0.000000,  0.016064, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.016064, -0.128482, -0.005602,  0.000000, -0.000000,  0.000000,  0.000000,
            -0.128482, -0.005602,  0.000000,  0.027270,  0.000000,  0.000000, -0.000000,  0.000000,
             0.043886, -0.000000,  0.000000,  0.027270, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.027270, -0.142758, -0.024138, -0.000000,  0.000000, -0.000000,  0.000000,
            -0.142758, -0.024138,  0.000000,  0.021883, -0.000000, -0.000000, -0.000000, -0.000000,
             0.000000,  0.021883,  0.062629, -0.000000, -0.000000, -0.000000,  0.000000, -0.000000,
             0.000000,  0.021883, -0.178447, -0.036606,  0.000000,  0.000000, -0.000000, -0.000000,
            -0.178447, -0.036606,  0.000000,  0.016064, -0.000000,  0.000000,  0.000000,  0.000000
        ]
    ]
    @test norm(mod.models[1].solution.z_curr[1:2]   .- Z_GEN[1][1:2],  Inf) <= atol # Gen 1 t=1
    @test norm(mod.models[1].solution.z_curr[3:4]   .- Z_GEN[1][3:4],  Inf) <= atol # Gen 2 t=1
    @test norm(mod.models[1].solution.z_curr[5:6]   .- Z_GEN[1][5:6],  Inf) <= atol # Gen 3 t=1
    @test norm(mod.models[2].solution.z_curr[1:2]   .- Z_GEN[2][1:2],  Inf) <= atol # Gen 1 t=2
    @test norm(mod.models[2].solution.z_curr[3:4]   .- Z_GEN[2][3:4],  Inf) <= atol # Gen 2 t=2
    @test norm(mod.models[2].solution.z_curr[5:6]   .- Z_GEN[2][5:6],  Inf) <= atol # Gen 3 t=2
    @test norm(mod.models[3].solution.z_curr[1:2]   .- Z_GEN[3][1:2],  Inf) <= atol # Gen 1 t=3
    @test norm(mod.models[3].solution.z_curr[3:4]   .- Z_GEN[3][3:4],  Inf) <= atol # Gen 2 t=3
    @test norm(mod.models[3].solution.z_curr[5:6]   .- Z_GEN[3][5:6],  Inf) <= atol # Gen 3 t=3
    @test norm(mod.models[1].solution.z_curr[7:14]  .- Z_BR[1][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[1].solution.z_curr[15:22] .- Z_BR[1][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[1].solution.z_curr[23:30] .- Z_BR[1][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[1].solution.z_curr[31:38] .- Z_BR[1][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[1].solution.z_curr[39:46] .- Z_BR[1][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[1].solution.z_curr[47:54] .- Z_BR[1][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[1].solution.z_curr[55:62] .- Z_BR[1][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[1].solution.z_curr[63:70] .- Z_BR[1][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[1].solution.z_curr[71:78] .- Z_BR[1][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[2].solution.z_curr[7:14]  .- Z_BR[2][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[2].solution.z_curr[15:22] .- Z_BR[2][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[2].solution.z_curr[23:30] .- Z_BR[2][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[2].solution.z_curr[31:38] .- Z_BR[2][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[2].solution.z_curr[39:46] .- Z_BR[2][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[2].solution.z_curr[47:54] .- Z_BR[2][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[2].solution.z_curr[55:62] .- Z_BR[2][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[2].solution.z_curr[63:70] .- Z_BR[2][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[2].solution.z_curr[71:78] .- Z_BR[2][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[3].solution.z_curr[7:14]  .- Z_BR[3][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[3].solution.z_curr[15:22] .- Z_BR[3][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[3].solution.z_curr[23:30] .- Z_BR[3][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[3].solution.z_curr[31:38] .- Z_BR[3][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[3].solution.z_curr[39:46] .- Z_BR[3][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[3].solution.z_curr[47:54] .- Z_BR[3][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[3].solution.z_curr[55:62] .- Z_BR[3][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[3].solution.z_curr[63:70] .- Z_BR[3][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[3].solution.z_curr[71:78] .- Z_BR[3][65:72], Inf) <= atol # Line 9 t=1

    # l update
    ExaAdmm.admm_update_l(env, mod)

    L_GEN = [
        [
            -2.222233, -0.000000,
            -2.115202, -0.000000,
            -3.656485, -0.000000
        ],
        [
             11.746018, -0.000000,
             36.038083, -0.000000,
             24.114272, -0.000000
        ],
        [
             24.761902, -0.000000,
             62.628569, -0.000000,
             43.885712, -0.000000
        ]
    ]
    L_BR = [
        [
              -30.793661,    -0.000000,    -0.000000,   -16.063810,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -16.063810,   128.571429,     5.631428,    -0.000000,     0.000000,    -0.000000,    -0.000000,
              128.571429,     5.631428,    -0.000000,   -27.270000,    -0.000000,    -0.000001,     0.000000,    -0.000000,
              -49.771773,    -0.000000,    -0.000000,   -27.270000,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -27.270000,   142.857143,    24.172857,     0.000000,    -0.000000,     0.000000,    -0.000000,
              142.857143,    24.172857,    -0.000000,   -21.883334,     0.000000,     0.000000,     0.000000,     0.000000,
               -0.000000,   -21.883334,   -70.142413,    -0.000000,     0.000000,    -0.000000,    -0.000000,    -0.000000,
               -0.000000,   -21.883334,   178.571429,    36.655714,    -0.000000,    -0.000000,     0.000000,     0.000000,
              178.571429,    36.655714,    -0.000000,   -16.063810,     0.000000,    -0.000000,    -0.000000,    -0.000000
        ],
        [
              -37.777787,    -0.000000,    -0.000000,   -16.063810,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -16.063810,   128.521323,     5.614726,    -0.000000,     0.000000,    -0.000000,    -0.000000,
              128.521323,     5.614726,    -0.000000,   -27.270000,    -0.000000,    -0.000001,     0.000000,    -0.000000,
              -63.657152,    -0.000000,    -0.000000,   -27.270000,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -27.270000,   142.801470,    24.153371,     0.000000,    -0.000000,     0.000000,    -0.000000,
              142.801470,    24.153371,    -0.000000,   -21.883334,     0.000000,     0.000000,     0.000000,     0.000000,
               -0.000000,   -21.883334,   -89.219055,    -0.000000,     0.000000,    -0.000000,    -0.000000,    -0.000000,
               -0.000000,   -21.883334,   178.501838,    36.627877,    -0.000000,    -0.000000,     0.000000,     0.000000,
              178.501838,    36.627877,    -0.000000,   -16.063810,     0.000000,    -0.000000,    -0.000000,    -0.000000
        ],
        [
              -24.761902,    -0.000000,    -0.000000,   -16.063810,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -16.063810,   128.481876,     5.601577,    -0.000000,     0.000000,    -0.000000,    -0.000000,
              128.481876,     5.601577,    -0.000000,   -27.270000,    -0.000000,    -0.000001,     0.000000,    -0.000000,
              -43.885712,    -0.000000,    -0.000000,   -27.270000,    -0.000000,     0.000000,    -0.000000,     0.000000,
               -0.000000,   -27.270000,   142.757640,    24.138031,     0.000000,    -0.000000,     0.000000,    -0.000000,
              142.757640,    24.138031,    -0.000000,   -21.883334,     0.000000,     0.000000,     0.000000,     0.000000,
               -0.000000,   -21.883334,   -62.628569,    -0.000000,     0.000000,    -0.000000,    -0.000000,    -0.000000,
               -0.000000,   -21.883334,   178.447050,    36.605962,    -0.000000,    -0.000000,     0.000000,     0.000000,
              178.447050,    36.605962,    -0.000000,   -16.063810,     0.000000,    -0.000000,    -0.000000,    -0.000000
        ]
    ]
    @test norm(mod.models[1].solution.l_curr[1:2]   .- L_GEN[1][1:2],  Inf) <= atol # Gen 1 t=1
    @test norm(mod.models[1].solution.l_curr[3:4]   .- L_GEN[1][3:4],  Inf) <= atol # Gen 2 t=1
    @test norm(mod.models[1].solution.l_curr[5:6]   .- L_GEN[1][5:6],  Inf) <= atol # Gen 3 t=1
    @test norm(mod.models[2].solution.l_curr[1:2]   .- L_GEN[2][1:2],  Inf) <= atol # Gen 1 t=2
    @test norm(mod.models[2].solution.l_curr[3:4]   .- L_GEN[2][3:4],  Inf) <= atol # Gen 2 t=2
    @test norm(mod.models[2].solution.l_curr[5:6]   .- L_GEN[2][5:6],  Inf) <= atol # Gen 3 t=2
    @test norm(mod.models[3].solution.l_curr[1:2]   .- L_GEN[3][1:2],  Inf) <= atol # Gen 1 t=3
    @test norm(mod.models[3].solution.l_curr[3:4]   .- L_GEN[3][3:4],  Inf) <= atol # Gen 2 t=3
    @test norm(mod.models[3].solution.l_curr[5:6]   .- L_GEN[3][5:6],  Inf) <= atol # Gen 3 t=3
    @test norm(mod.models[1].solution.l_curr[7:14]  .- L_BR[1][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[1].solution.l_curr[15:22] .- L_BR[1][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[1].solution.l_curr[23:30] .- L_BR[1][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[1].solution.l_curr[31:38] .- L_BR[1][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[1].solution.l_curr[39:46] .- L_BR[1][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[1].solution.l_curr[47:54] .- L_BR[1][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[1].solution.l_curr[55:62] .- L_BR[1][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[1].solution.l_curr[63:70] .- L_BR[1][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[1].solution.l_curr[71:78] .- L_BR[1][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[2].solution.l_curr[7:14]  .- L_BR[2][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[2].solution.l_curr[15:22] .- L_BR[2][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[2].solution.l_curr[23:30] .- L_BR[2][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[2].solution.l_curr[31:38] .- L_BR[2][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[2].solution.l_curr[39:46] .- L_BR[2][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[2].solution.l_curr[47:54] .- L_BR[2][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[2].solution.l_curr[55:62] .- L_BR[2][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[2].solution.l_curr[63:70] .- L_BR[2][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[2].solution.l_curr[71:78] .- L_BR[2][65:72], Inf) <= atol # Line 9 t=1
    @test norm(mod.models[3].solution.l_curr[7:14]  .- L_BR[3][1:8],   Inf) <= atol # Line 1 t=1
    @test norm(mod.models[3].solution.l_curr[15:22] .- L_BR[3][9:16],  Inf) <= atol # Line 2 t=1
    @test norm(mod.models[3].solution.l_curr[23:30] .- L_BR[3][17:24], Inf) <= atol # Line 3 t=1
    @test norm(mod.models[3].solution.l_curr[31:38] .- L_BR[3][25:32], Inf) <= atol # Line 4 t=1
    @test norm(mod.models[3].solution.l_curr[39:46] .- L_BR[3][33:40], Inf) <= atol # Line 5 t=1
    @test norm(mod.models[3].solution.l_curr[47:54] .- L_BR[3][41:48], Inf) <= atol # Line 6 t=1
    @test norm(mod.models[3].solution.l_curr[55:62] .- L_BR[3][49:56], Inf) <= atol # Line 7 t=1
    @test norm(mod.models[3].solution.l_curr[63:70] .- L_BR[3][57:64], Inf) <= atol # Line 8 t=1
    @test norm(mod.models[3].solution.l_curr[71:78] .- L_BR[3][65:72], Inf) <= atol # Line 9 t=1

    # residual update
    ExaAdmm.admm_update_residual(env, mod)

    for i=1:mod.len_horizon
        sol = mod.models[i].solution
        @test norm(sol.rp .- (sol.u_curr .- sol.v_curr .+ sol.z_curr), Inf) <= atol
        @test norm(sol.rd .- (sol.z_curr .- sol.z_prev),               Inf) <= atol
        @test norm(sol.Ax_plus_By .- (sol.u_curr .- sol.v_curr),       Inf) <= atol
        if i > 1
            sol = mod.solution[i]
            submod = mod.models[i]
            v_curr = @view mod.models[i-1].solution.v_curr[submod.gen_start:2:submod.gen_start+2*submod.ngen-1]
            @test norm(sol.rp .- (sol.u_curr .- v_curr .+ sol.z_curr), Inf) <= atol
            @test norm(sol.rd .- (sol.z_curr .- sol.z_prev),               Inf) <= atol
            @test norm(sol.Ax_plus_By .- (sol.u_curr .- v_curr),       Inf) <= atol
        end
    end

    # lz update
    lz_prev = Vector{Vector{Float64}}(undef, mod.len_horizon)
    lz_ramp_prev = Vector{Vector{Float64}}(undef, mod.len_horizon)
    for i=1:mod.len_horizon
        lz_prev[i] = zeros(length(mod.models[i].solution.lz))
        lz_prev[i] .= mod.models[i].solution.lz
        lz_ramp_prev[i] = zeros(length(mod.solution[i].lz))
        lz_ramp_prev[i] .= mod.solution[i].lz
    end

    ExaAdmm.admm_update_lz(env, mod)

    for i=1:mod.len_horizon
        sol = mod.models[i].solution
        @test norm(sol.lz .- (lz_prev[i] .+ (env.params.beta .* sol.z_curr)), Inf) <= atol
        if i > 1
            sol = mod.solution[i]
            @test norm(sol.lz .- (lz_ramp_prev[i] .+ (env.params.beta .* sol.z_curr)), Inf) <= atol
        end
    end

    env, mod = ExaAdmm.solve_mpacopf(case, load_prefix; end_period=end_period, warm_start=false, outer_iterlim=25, rho_pq=rho_pq, rho_va=rho_va, outer_eps=2*1e-5, verbose=verbose)
    @test mod.info.status == :Solved
    @test mod.info.outer == 20
    @test mod.info.cumul == 729
    @test isapprox(mod.info.objval, 15901.48; atol=1e-2)

#=
    @printf("L_GEN = [\n")
    for i=1:mod.len_horizon
        @printf("    [\n")
        for g=1:mod.models[i].ngen
            pg_idx = mod.models[i].gen_start + 2*(g-1)
            @printf("        % .6f, % .6f", mod.models[i].solution.l_curr[pg_idx], mod.models[i].solution.l_curr[pg_idx+1])
            if g < mod.models[i].ngen
                @printf(",\n")
            else
                @printf("\n")
            end
        end
        if i < mod.len_horizon
            @printf("    ],\n")
        else
            @printf("    ]\n")
        end
    end
    @printf("]\n")

    @printf("L_BR = [\n")
    for i=1:mod.len_horizon
        @printf("    [\n")
        for l=1:mod.models[i].nline
            @printf("        % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f, % 12.6f",
                    mod.models[i].solution.l_curr[7+8*(l-1)],
                    mod.models[i].solution.l_curr[7+8*(l-1)+1],
                    mod.models[i].solution.l_curr[7+8*(l-1)+2],
                    mod.models[i].solution.l_curr[7+8*(l-1)+3],
                    mod.models[i].solution.l_curr[7+8*(l-1)+4],
                    mod.models[i].solution.l_curr[7+8*(l-1)+5],
                    mod.models[i].solution.l_curr[7+8*(l-1)+6],
                    mod.models[i].solution.l_curr[7+8*(l-1)+7]
            )
            if l < mod.models[i].nline
                @printf(",\n")
            else
                @printf("\n")
            end
        end
        if i < mod.len_horizon
            @printf("    ],\n")
        else
            @printf("    ]\n")
        end
    end
    @printf("]\n")
=#

end