# testing Tron functionality on GPU and learn Array Programming 

using ExaAdmm
using SparseArrays
using Ipopt
using JuMP
using LinearAlgebra


#solve QP 1/2 x^T*A*x + b*x
A_temp = rand(6,6)

A_cpu = transpose(A_temp)*A_temp #symmetric and PSD
b_cpu = rand(6)
l_cpu = rand(6)
u_cpu = l_cpu .+ 20
x0_cpu = 0.5(l_cpu + u_cpu)

function eval_f_cpu(x) #f cpu
    return 0.5*dot(x,A_cpu,x) + dot(b_cpu, x)
end


function eval_g_cpu(x, g) 
    fill!(g, 0)
    mul!(g, A_cpu, x)
    g .+= b_cpu
end



function eval_h_cpu(x, mode, rows, cols, obj_factor, lambda, values)
    if mode == :Structure
        # for i in 1:nnz(P)
        #     rows[i] = Iz[i]
        #     cols[i] = Jz[i]
        # end
        nz = 1
        for j=1:6
            for i in j:6 
                rows[nz] = i
                cols[nz] = j
                nz += 1
            end
        end
    else
        # copy!(values, vals)
        nz = 1
        for j=1:6
            for i in j:6 
                values[nz] = A_cpu[i, j]
                nz += 1
            end
        end
    end
end

println("QP validation: IPOPT and Tron GPU")
#solve with Ipopt
model = JuMP.Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, l_cpu[i]<= x[i=1:6] <=u_cpu[i])
@objective(model, Min, 0.5 * dot(x, A_cpu, x) + dot(b_cpu, x))
optimize!(model)
println("IPOPT: result x = ",value.(x), " with objective = ", objective_value(model))


Tron_cpu = ExaAdmm.ExaTron.createProblem(6, l_cpu, u_cpu, 21, eval_f_cpu, eval_g_cpu, eval_h_cpu; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12)
Tron_cpu.x = x0_cpu #initialization 
status_tron_cpu = ExaAdmm.ExaTron.solveProblem(Tron_cpu)
println("Tron cpu: result x = ", Tron_cpu.x, " with objective = ", Tron_cpu.f, "with multiplier =", Tron_cpu.g)

#! indexing error
# Tron_gpu = ExaAdmm.ExaTron.createProblem(6, l, u, 21, eval_f, eval_g, eval_h; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12)
# status_tron_gpu = ExaAdmm.ExaTron.solveProblem(Tron_gpu)
# println("Tron gpu: result x = ", Tron_gpu.x, " with objective = ", Tron_gpu.f)

# #test 1 toy example sol = l + u testing threads cuda syntax
# function test1(l, u, sol)
#     idx = blockIdx().x
#     tx = threadIdx().x
#     if tx <= 6
#         # @cuprintf("here\n")
#         sol[tx] = l[tx] + u[tx]
#     end
#     return 
# end 

# result = CuArray(zeros(6))
# @cuda threads=6 blocks=1 test1(l,u,result)

#test 2 toy example eval_f testing shmem and function return and kernel vector copy 
# f_sol = CuArray(zeros(1))
# x_sol = CuArray(zeros(6))

# # shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)
# shmem_size = sizeof(Float64)*6
using CUDA

gpu_no = 1
CUDA.device!(gpu_no)

A = CuArray(A_cpu)
b = CuArray(b_cpu)
l = CuArray(l_cpu)
u = CuArray(u_cpu)
x0 = CuArray(x0_cpu)


function eval_f_kernel(x::CuDeviceArray{Float64,1},A::CuDeviceArray{Float64,2},b::CuDeviceArray{Float64,1}) #f gpu
    f = 0.0
    tx = threadIdx().x
    # if tx <= 1
    #TODO: expand if needed
        @inbounds begin
            for i = 1:6
                for j = 1:6
                    f += 0.5*x[i]*A[i,j]*x[j]
                end
                    f += b[i]*x[i]
            end
        end
    # end
    CUDA.sync_threads()
    return f 
end

function eval_g_kernel(x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2}, b::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    # fill!(g, 0.0)
    # @cuprintf("  g1 = %.16e\n", g[1])
    # @cuprintf("  g2 = %.16e\n", g[2])
    # @cuprintf("  g3 = %.16e\n", g[3])
    # @cuprintf("  g4 = %.16e\n", g[4])
    # @cuprintf("  g5 = %.16e\n", g[5])
    # @cuprintf("  g6 = %.16e\n", g[6])
    g1 = A[1,1]*x[1] + A[1,2]*x[2] + A[1,3]*x[3] + A[1,4]*x[4] + A[1,5]*x[5] + A[1,6]*x[6] + b[1]
    g2 = A[2,1]*x[1] + A[2,2]*x[2] + A[2,3]*x[3] + A[2,4]*x[4] + A[2,5]*x[5] + A[2,6]*x[6] + b[2]
    g3 = A[3,1]*x[1] + A[3,2]*x[2] + A[3,3]*x[3] + A[3,4]*x[4] + A[3,5]*x[5] + A[3,6]*x[6] + b[3]
    g4 = A[4,1]*x[1] + A[4,2]*x[2] + A[4,3]*x[3] + A[4,4]*x[4] + A[4,5]*x[5] + A[4,6]*x[6] + b[4]
    g5 = A[5,1]*x[1] + A[5,2]*x[2] + A[5,3]*x[3] + A[5,4]*x[4] + A[5,5]*x[5] + A[5,6]*x[6] + b[5]
    g6 = A[6,1]*x[1] + A[6,2]*x[2] + A[6,3]*x[3] + A[6,4]*x[4] + A[6,5]*x[5] + A[6,6]*x[6] + b[6]
    if tx <= 1
        g[1] = g1
        g[2] = g2
        g[3] = g3
        g[4] = g4
        g[5] = g5
        g[6] = g6
    #     @cuprintf("  g1 = %.16e\n", g[1])
    # @cuprintf("  g2 = %.16e\n", g[2])
    # @cuprintf("  g3 = %.16e\n", g[3])
    # @cuprintf("  g4 = %.16e\n", g[4])
    # @cuprintf("  g5 = %.16e\n", g[5])
    # @cuprintf("  g6 = %.16e\n", g[6])
    end
    CUDA.sync_threads()
end

function eval_h_kernel(A::CuDeviceArray{Float64,2}, H::CuDeviceArray{Float64,2})
    tx = threadIdx().x
    #TODO: expand if needed
    if tx <= 1
        @inbounds begin
            for i = 1:6
                for j = 1:6
                    A[i,j] = H[i,j]
                end
            end
        end
    end
    CUDA.sync_threads()
end

# function test2(A,b,f_sol,x_sol)
#     tx = threadIdx().x

#     x_tmp = CuDynamicSharedArray(Float64, 6)
#     x_tmp[1] = 1
#     x_tmp[2] = 2
#     x_tmp[3] = 3
#     x_tmp[4] = 4
#     x_tmp[5] = 5
#     x_tmp[6] = 6

#     # f_sol[1] = 1 #x_sol[5]

#     if tx <= 1
#         @cuprintf("test2: loc 1: tx = %.16e\n", tx)
#         # f_sol[1] = x_sol[5]
#         f_sol[1] = eval_f_kernel(x_tmp, A, b)
#         x_sol[1] = x_tmp[1]
#         x_sol[2] = x_tmp[2]
#         x_sol[3] = x_tmp[3]
#         x_sol[4] = x_tmp[4]
#         x_sol[5] = x_tmp[5]
#         x_sol[6] = x_tmp[6]
#     end
#     CUDA.sync_threads()
#     return 
# end

# @cuda threads=1 blocks=1 shmem=shmem_size test2(A,b,f_sol,x_sol)

#tron gpu solver test 
n = 6
# f_sol2 = CuArray(zeros(1))
x_sol = CuArray(zeros(6))
g_sol = CuArray(zeros(6))
shmem_size2 = sizeof(Float64)*(14*n+3*n^2) + sizeof(Int)*(4*n)

function gpu_test(n::Int,H::CuDeviceArray{Float64,2},b::CuDeviceArray{Float64,1},l::CuDeviceArray{Float64,1},u::CuDeviceArray{Float64,1},x_sol::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    x = CuDynamicSharedArray(Float64, n) #memory allocation 
    xl = CuDynamicSharedArray(Float64, n, n*sizeof(Float64))
    xu = CuDynamicSharedArray(Float64, n, (2*n)*sizeof(Float64))

        xl[1] = l[1]
        xl[2] = l[2]
        xl[3] = l[3]
        xl[4] = l[4]
        xl[5] = l[5]
        xl[6] = l[6]
        xu[1] = u[1]
        xu[2] = u[2]
        xu[3] = u[3]
        xu[4] = u[4]
        xu[5] = u[5]
        xu[6] = u[6]
        x[1] = (xl[1] + xu[1])/2
        x[2] = (xl[2] + xu[2])/2
        x[3] = (xl[3] + xu[3])/2
        x[4] = (xl[4] + xu[4])/2
        x[5] = (xl[5] + xu[5])/2
        x[6] = (xl[6] + xu[6])/2

    # if tx <= 1
        

        # @cuprintf("  x1 = %.16e\n", x[1])
        #     @cuprintf("  x2 = %.16e\n", x[2])
        #     @cuprintf("  x3 = %.16e\n", x[3])
        #     @cuprintf("  x4 = %.16e\n", x[4])
        #     @cuprintf("  x5 = %.16e\n", x[5])
        #     @cuprintf("  x6 = %.16e\n", x[6])

           tron_gpu_test(n, H, b, x, xl, xu)

        #    if tx == 1
                x_sol[1] = x[1]
                x_sol[2] = x[2]
                x_sol[3] = x[3]
                x_sol[4] = x[4]
                x_sol[5] = x[5]
                x_sol[6] = x[6]

                # g_sol[1] = g[1]
                # g_sol[2] = g[2]
                # g_sol[3] = g[3]
                # g_sol[4] = g[4]
                # g_sol[5] = g[5]
                # g_sol[6] = g[6]
        #    end
    # end
    CUDA.sync_threads()
    return 
end

function tron_gpu_test(n::Int, H::CuDeviceArray{Float64,2}, b::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1})
    tx = threadIdx().x

    

    #? shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n) where n = 6
    g = CuDynamicSharedArray(Float64, n, (3*n)*sizeof(Float64))
    xc = CuDynamicSharedArray(Float64, n, (4*n)*sizeof(Float64))
    s = CuDynamicSharedArray(Float64, n, (5*n)*sizeof(Float64))
    wa = CuDynamicSharedArray(Float64, n, (6*n)*sizeof(Float64))
    wa1 = CuDynamicSharedArray(Float64, n, (7*n)*sizeof(Float64))
    wa2 = CuDynamicSharedArray(Float64, n, (8*n)*sizeof(Float64))
    wa3 = CuDynamicSharedArray(Float64, n, (9*n)*sizeof(Float64))
    wa4 = CuDynamicSharedArray(Float64, n, (10*n)*sizeof(Float64))
    wa5 = CuDynamicSharedArray(Float64, n, (11*n)*sizeof(Float64))
    gfree = CuDynamicSharedArray(Float64, n, (12*n)*sizeof(Float64))
    dsave = CuDynamicSharedArray(Float64, n, (13*n)*sizeof(Float64))
    indfree = CuDynamicSharedArray(Int, n, (14*n)*sizeof(Float64))
    iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (14*n)*sizeof(Float64))
    isave = CuDynamicSharedArray(Int, n, (3*n)*sizeof(Int) + (14*n)*sizeof(Float64))

    # @cuprint(x[1,1,1,1])

    A = CuDynamicSharedArray(Float64, (n,n), (14*n)*sizeof(Float64)+(4*n)*sizeof(Int))
    B = CuDynamicSharedArray(Float64, (n,n), (14*n+n^2)*sizeof(Float64)+(4*n)*sizeof(Int))
    L = CuDynamicSharedArray(Float64, (n,n), (14*n+2*n^2)*sizeof(Float64)+(4*n)*sizeof(Int))

    #? what for 
    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    CUDA.sync_threads()

    # xl[1] = l[1]
    # xl[2] = l[2]
    # xl[3] = l[3]
    # xl[4] = l[4]
    # xl[5] = l[5]
    # xl[6] = l[6]
    # xu[1] = u[1]
    # xu[2] = u[2]
    # xu[3] = u[3]
    # xu[4] = u[4]
    # xu[5] = u[5]
    # xu[6] = u[6]
    # x[1] = (xl[1] + xu[1])/2
    # x[2] = (xl[2] + xu[2])/2
    # x[3] = (xl[3] + xu[3])/2
    # x[4] = (xl[4] + xu[4])/2
    # x[5] = (xl[5] + xu[5])/2
    # x[6] = (xl[6] + xu[6])/2

            # @cuprintf("  x1 = %.16e\n", x[1])
            # @cuprintf("  x2 = %.16e\n", x[2])
            # @cuprintf("  x3 = %.16e\n", x[3])
            # @cuprintf("  x4 = %.16e\n", x[4])
            # @cuprintf("  x5 = %.16e\n", x[5])
            # @cuprintf("  x6 = %.16e\n", x[6])

    max_feval = 500
    max_minor = 200
    gtol = 1e-6

    task = 0
    status = 0

    delta = 0.0
    fatol = 0.0
    frtol = 1e-12
    fmin = -1e32
    cgtol = 0.1
    cg_itermax = n

    f = 0.0
    nfev = 0
    ngev = 0
    nhev = 0
    minor_iter = 0
    search = true

    # #? for debug
    # f_sol2[1] = 1.0
    # x_sol2[1] = 1.0
    while search
# # #=
# #         if threadIdx().x == 1
# #             @cuprintf("iter = %d\n", minor_iter)
# #             for i=1:n
# #                 @cuprintf("  x[%d] = %.16e\n", i, x[i])
# #             end
# #         end
# # =#
#         # [0|1]: Evaluate function.

        if task == 0 || task == 1
            # @cuprintf("  f = %.16e\n", f)
            f = eval_f_kernel(x,H,b)
            # @cuprintf("  f = %.16e\n", f)
            # @cuprintf(typeof(f))
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

# #         # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_g_kernel(x,g,H,b)
            eval_h_kernel(A,H)
            # @cuprintf("  g1 = %.16e\n", g[1])
            # @cuprintf("  g2 = %.16e\n", g[2])
            # @cuprintf("  g3 = %.16e\n", g[3])
            # @cuprintf("  g4 = %.16e\n", g[4])
            # @cuprintf("  g5 = %.16e\n", g[5])
            # @cuprintf("  g6 = %.16e\n", g[6])
            # @cuprintf("  A11 = %.16e\n", A[1,1])
            # @cuprintf("  A22 = %.16e\n", A[2,2])
            # @cuprintf("  A33 = %.16e\n", A[3,3])
            # @cuprintf("  A44 = %.16e\n", A[4,4])
            # @cuprintf("  A55 = %.16e\n", A[5,5])
            # @cuprintf("  A66 = %.16e\n", A[6,6])
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

# #         # Initialize the trust region bound.

        if task == 0
            gnorm0 = ExaAdmm.ExaTron.dnrm2(n, g, 1)
            delta = gnorm0 
            # @cuprintf("  delta = %.16e\n", delta)
            # @cuprintf("  gnorm0 = %.16e\n", gnorm0)
        end

# #         # Call Tron.
# # #=
#         if tx == 1
#             @cuprintln("minor_iter = ", minor_iter, " task = ", task, " f = ", f)
#         end
# # =#
        if search
            # @cuprintln(typeof(g))
            # @cushow(typeof(g))
            # @cuprintf("%s",string(typeof(g)))
            # typ = typeof(f)
            delta, task = ExaAdmm.ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                        isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
            # @cuprintf("  delta = %.16e\n", delta)
        end

# #         # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaAdmm.ExaTron.dgpnorm(n, x, xl, xu, g)
#=
            if tx == 1
                @cuprintln("  gnorm_inf = ", gnorm_inf)
            end
=#
            if gnorm_inf <= gtol
                task = 4
            end

            if minor_iter >= max_minor
                status = 1
                search = false
            end
        end

# #         # [4] CONV: convergence was achieved.
# #         # [10] : warning fval is less than fmin

        if task == 4 || task == 10
            search = false
        end
        # search = false #? for debug
     end # end while 
    
     
    CUDA.sync_threads()
    # f_sol2[1] = f
    # if tx == 1
    #     @cuprintf("  x1 = %.16e\n", x[1])
    #         @cuprintf("  x2 = %.16e\n", x[2])
    #         @cuprintf("  x3 = %.16e\n", x[3])
    #         @cuprintf("  x4 = %.16e\n", x[4])
    #         @cuprintf("  x5 = %.16e\n", x[5])
    #         @cuprintf("  x6 = %.16e\n", x[6]) 
    # end

    # @cuprintf("  g1 = %.16e\n", g[1])
    #         @cuprintf("  g2 = %.16e\n", g[2])
    #         @cuprintf("  g3 = %.16e\n", g[3])
    #         @cuprintf("  g4 = %.16e\n", g[4])
    #         @cuprintf("  g5 = %.16e\n", g[5])
    #         @cuprintf("  g6 = %.16e\n", g[6])

    return
end

# function test(n::Int,A::CuArray{Float64,2}, b::CuArray{Float64,1}, l::CuArray{Float64,1}, u::CuArray{Float64,1})
#     return 
# end

@cuda threads=32 blocks=1 shmem=shmem_size2 gpu_test(n,A,b,l,u,x_sol)

# println("IPOPT: result x = ",value.(x), " with objective = ", objective_value(model))