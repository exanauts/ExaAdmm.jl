using KernelAbstractions
using CUDA
using AMDGPU
KA = KernelAbstractions
devices = Vector{KA.Device}()
push!(devices, KA.CPU())
if CUDA.has_cuda_gpu() || AMDGPU.has_rocm_gpu()
    if CUDA.has_cuda_gpu()
        using CUDAKernels
        push!(devices, CUDADevice())
    end
    if AMDGPU.has_rocm_gpu()
        using ROCKernels
        push!(devices, ROCDevice())
    end
end
@testset "Testing [x,xbar,z,l,lz] updates and a solve for ACOPF using KA" for _device in devices
    # Testing case9.m
    case = joinpath(INSTANCES_DIR, "case9.m")
    rho_pq = 4e2; rho_va = 4e4
    atol = 1e-6; verbose=0
    device = _device

    # Initialize an ACOPF model with default options.
    T = Float64
    if isa(device, KA.CPU)
        TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
        env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; use_gpu=false, ka_device=device, verbose=verbose)
    else isa(device, KA.GPU)
        if isa(device, CUDADevice)
            TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
        elseif isa(device, ROCDevice)
            TD = ROCArray{Float64,1}; TI = ROCArray{Int,1}; TM = ROCArray{Float64,2}
        else
            error("Unsupported device $device")
        end
        env = ExaAdmm.AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; use_gpu=true, ka_device=device, verbose=verbose)
    end
    mod = ExaAdmm.ModelAcopf{T,TD,TI,TM}(env)
    device = isa(_device, KA.CPU) ? nothing : _device

    sol = mod.solution

    env.params.scale = 1e-4
    env.params.obj_scale = 1.0
    env.params.initial_beta = 1e3
    env.params.beta = 1e3

    # # We'll perform just one ADMM iteration. Do some pre-steps first.
    ExaAdmm.admm_increment_outer(env, mod, device)
    ExaAdmm.admm_outer_prestep(env, mod, device)
    ExaAdmm.admm_increment_reset_inner(env, mod, device)
    ExaAdmm.admm_increment_inner(env, mod, device)
    ExaAdmm.admm_inner_prestep(env, mod, device)

    # x update
    ExaAdmm.admm_update_x(env, mod, device)

    # Referece solution. Entries are in the order of their numbers.
    U_GEN = [
            0.1, 0.0,
        0.238095, 0.0,
        0.161403, 0.0
    ]
    U_BR  = [
        0.0,       0.0, 0.0,       0.0, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.079790, 0.0, -0.079790, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.180790, 0.0, -0.180790, 1.01, 1.01, 0.0, 0.0,
        0.0,       0.0, 0.0,       0.0, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.105545, 0.0, -0.105545, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.075245, 0.0, -0.075245, 1.01, 1.01, 0.0, 0.0,
        0.0,       0.0, 0.0,       0.0, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.154530, 0.0, -0.154530, 1.01, 1.01, 0.0, 0.0,
        0.0, -0.088880, 0.0, -0.088880, 1.01, 1.01, 0.0, 0.0
    ]
    u_curr = zeros(length(sol.u_curr))
    copyto!(u_curr, sol.u_curr)
    @test norm(u_curr[1:2]   .- U_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(u_curr[3:4]   .- U_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(u_curr[5:6]   .- U_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(u_curr[7:14]  .- U_BR[1:8],   Inf) <= atol # Line 1
    @test norm(u_curr[15:22] .- U_BR[9:16],  Inf) <= atol # Line 2
    @test norm(u_curr[23:30] .- U_BR[17:24], Inf) <= atol # Line 3
    @test norm(u_curr[31:38] .- U_BR[25:32], Inf) <= atol # Line 4
    @test norm(u_curr[39:46] .- U_BR[33:40], Inf) <= atol # Line 5
    @test norm(u_curr[47:54] .- U_BR[41:48], Inf) <= atol # Line 6
    @test norm(u_curr[55:62] .- U_BR[49:56], Inf) <= atol # Line 7
    @test norm(u_curr[63:70] .- U_BR[57:64], Inf) <= atol # Line 8
    @test norm(u_curr[71:78] .- U_BR[65:72], Inf) <= atol # Line 9

    # xbar update
    ExaAdmm.admm_update_xbar(env, mod, device)

    V_GEN = [
            0.05, 0.0,
        0.119047, 0.0,
        0.080701, 0.0
    ]
    V_BR  = [
            0.05,       0.0,      0.0,  0.056223, 1.01, 1.01, 0.0, 0.0,
            0.0, -0.023566,    -0.45, -0.099499, 1.01, 1.01, 0.0, 0.0,
        -0.45, -0.200500,      0.0, -0.085345, 1.01, 1.01, 0.0, 0.0,
        0.080701,       0.0,      0.0,  0.095445, 1.01, 1.01, 0.0, 0.0,
            0.0, -0.010100,     -0.5, -0.190150, 1.01, 1.01, 0.0, 0.0,
            -0.5, -0.159849,      0.0,  0.001346, 1.01, 1.01, 0.0, 0.0,
            0.0,  0.076591, 0.119047,       0.0, 1.01, 1.01, 0.0, 0.0,
            0.0, -0.077938,   -0.625, -0.282825, 1.01, 1.01, 0.0, 0.0,
        -0.625, -0.217174,      0.0, -0.032656, 1.01, 1.01, 0.0, 0.0
    ]
    v_curr = zeros(length(sol.v_curr))
    copyto!(v_curr, sol.v_curr)

    @test norm(v_curr[1:2]   .- V_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(v_curr[3:4]   .- V_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(v_curr[5:6]   .- V_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(v_curr[1:6]   .- V_GEN,       Inf) <= atol
    @test norm(v_curr[7:14]  .- V_BR[1:8],   Inf) <= atol
    @test norm(v_curr[15:22] .- V_BR[9:16],  Inf) <= atol
    @test norm(v_curr[23:30] .- V_BR[17:24], Inf) <= atol
    @test norm(v_curr[31:38] .- V_BR[25:32], Inf) <= atol
    @test norm(v_curr[39:46] .- V_BR[33:40], Inf) <= atol
    @test norm(v_curr[47:54] .- V_BR[41:48], Inf) <= atol
    @test norm(v_curr[55:62] .- V_BR[49:56], Inf) <= atol
    @test norm(v_curr[63:70] .- V_BR[57:64], Inf) <= atol
    @test norm(v_curr[71:78] .- V_BR[65:72], Inf) <= atol

    # z update
    ExaAdmm.admm_update_z(env, mod, device)

    Z_GEN = [
        -0.014285, 0.0,
        -0.034013, 0.0,
        -0.023057, 0.0
    ]
    Z_BR  = [
        0.014285,        0.0,       0.0,  0.016063, 0.0, 0.0, 0.0, 0.0,
        0.0,        0.016063, -0.128571, -0.005631, 0.0, 0.0, 0.0, 0.0,
        -0.128571, -0.005631,       0.0,  0.027270, 0.0, 0.0, 0.0, 0.0,
        0.023057,        0.0,       0.0,  0.027270, 0.0, 0.0, 0.0, 0.0,
        0.0,        0.027270, -0.142857, -0.024172, 0.0, 0.0, 0.0, 0.0,
        -0.142857, -0.024172,       0.0,  0.021883, 0.0, 0.0, 0.0, 0.0,
        0.0,        0.021883,  0.034013,       0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,        0.021883, -0.178571, -0.036655, 0.0, 0.0, 0.0, 0.0,
        -0.178571, -0.036655,       0.0,  0.016063, 0.0, 0.0, 0.0, 0.0
    ]
    z_curr = zeros(length(sol.z_curr))
    z_prev = zeros(length(sol.z_prev))
    copyto!(z_curr, sol.z_curr)
    copyto!(z_prev, sol.z_prev)
    @test norm(z_curr[1:2]   .- Z_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(z_curr[3:4]   .- Z_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(z_curr[5:6]   .- Z_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(z_curr[7:14]  .- Z_BR[1:8],   Inf) <= atol # Line 1
    @test norm(z_curr[15:22] .- Z_BR[9:16],  Inf) <= atol # Line 2
    @test norm(z_curr[23:30] .- Z_BR[17:24], Inf) <= atol # Line 3
    @test norm(z_curr[31:38] .- Z_BR[25:32], Inf) <= atol # Line 4
    @test norm(z_curr[39:46] .- Z_BR[33:40], Inf) <= atol # Line 5
    @test norm(z_curr[47:54] .- Z_BR[41:48], Inf) <= atol # Line 6
    @test norm(z_curr[55:62] .- Z_BR[49:56], Inf) <= atol # Line 7
    @test norm(z_curr[63:70] .- Z_BR[57:64], Inf) <= atol # Line 8
    @test norm(z_curr[71:78] .- Z_BR[65:72], Inf) <= atol # Line 9

    # l update
    ExaAdmm.admm_update_l(env, mod, device)

    L_GEN = [
        14.285714, 0.0,
        34.013605, 0.0,
        23.057644, 0.0
    ]
    L_BR  = [
        -14.285714,        0.0,        0.0, -16.063809, 0.0, 0.0, 0.0, 0.0,
            0.0, -16.063809, 128.571428,   5.631428, 0.0, 0.0, 0.0, 0.0,
        128.571428,   5.631428,        0.0, -27.270000, 0.0, 0.0, 0.0, 0.0,
        -23.057644,        0.0,        0.0, -27.270000, 0.0, 0.0, 0.0, 0.0,
            0.0, -27.270000, 142.857142,  24.172856, 0.0, 0.0, 0.0, 0.0,
        142.857142,  24.172856,        0.0, -21.883333, 0.0, 0.0, 0.0, 0.0,
            0.0, -21.883333, -34.013605,        0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, -21.883333, 178.571428,  36.655713, 0.0, 0.0, 0.0, 0.0,
        178.571428,  36.655713,        0.0, -16.063809, 0.0, 0.0, 0.0, 0.0
    ]
    l_curr = zeros(length(sol.l_curr))
    copyto!(l_curr, sol.l_curr)
    @test norm(l_curr[1:2]   .- L_GEN[1:2],  Inf) <= atol # Gen 1
    @test norm(l_curr[3:4]   .- L_GEN[3:4],  Inf) <= atol # Gen 2
    @test norm(l_curr[5:6]   .- L_GEN[5:6],  Inf) <= atol # Gen 3
    @test norm(l_curr[7:14]  .- L_BR[1:8],   Inf) <= atol # Line 1
    @test norm(l_curr[15:22] .- L_BR[9:16],  Inf) <= atol # Line 2
    @test norm(l_curr[23:30] .- L_BR[17:24], Inf) <= atol # Line 3
    @test norm(l_curr[31:38] .- L_BR[25:32], Inf) <= atol # Line 4
    @test norm(l_curr[39:46] .- L_BR[33:40], Inf) <= atol # Line 5
    @test norm(l_curr[47:54] .- L_BR[41:48], Inf) <= atol # Line 6
    @test norm(l_curr[55:62] .- L_BR[49:56], Inf) <= atol # Line 7
    @test norm(l_curr[63:70] .- L_BR[57:64], Inf) <= atol # Line 8
    @test norm(l_curr[71:78] .- L_BR[65:72], Inf) <= atol # Line 9

    # residual update
    ExaAdmm.admm_update_residual(env, mod, device)

    rp = zeros(length(sol.rp))
    rd = zeros(length(sol.rd))
    Ax_plus_By = zeros(length(sol.Ax_plus_By))
    copyto!(rp, sol.rp)
    copyto!(rd, sol.rd)
    copyto!(Ax_plus_By, sol.Ax_plus_By)
    @test norm(rp .- (u_curr .- v_curr .+ z_curr), Inf) <= atol
    @test norm(rd .- (z_curr .- z_prev),           Inf) <= atol
    @test norm(Ax_plus_By .- (u_curr .- v_curr),   Inf) <= atol

    # lz update
    lz_prev = zeros(length(sol.lz))
    copyto!(lz_prev, sol.lz)

    ExaAdmm.admm_update_lz(env, mod, device)

    lz = zeros(length(sol.lz))
    copyto!(lz, sol.lz)

    @test norm(lz .- (lz_prev .+ (env.params.beta .* z_curr)), Inf) <= atol
    use_gpu = isa(_device, KA.CPU) ? false : true
    env, mod = ExaAdmm.solve_acopf(case; outer_iterlim=25, use_gpu=use_gpu, ka_device=_device, rho_pq=rho_pq, rho_va=rho_va, outer_eps=2*1e-5, verbose=verbose)
    @test mod.info.status == :Solved
    @test mod.info.outer == 20
    @test mod.info.cumul == 705
    @test isapprox(mod.info.objval, 5303.435; atol=1e-3)
end
