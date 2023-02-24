"""
Driver to run TRON on GPU. This should be called from a kernel.

    tron_gpu_test()
- solves bounded QP: 1/2 x^THx + b*x s.t xl <= x <= xu
- includes eval_f_kernel(), eval_g_kernel(), and eval_h_kernel
"""
@inline function tron_gpu_test(n::Int, H::CuDeviceArray{Float64,2}, b::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1})
    tx = threadIdx().x
    

    g = CuDynamicSharedArray(Float64, n, (3*n + (2*n + n^2 + 178))*sizeof(Float64))
    xc = CuDynamicSharedArray(Float64, n, (4*n + (2*n + n^2 + 178))*sizeof(Float64))
    s = CuDynamicSharedArray(Float64, n, (5*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa = CuDynamicSharedArray(Float64, n, (6*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa1 = CuDynamicSharedArray(Float64, n, (7*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa2 = CuDynamicSharedArray(Float64, n, (8*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa3 = CuDynamicSharedArray(Float64, n, (9*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa4 = CuDynamicSharedArray(Float64, n, (10*n + (2*n + n^2 + 178))*sizeof(Float64))
    wa5 = CuDynamicSharedArray(Float64, n, (11*n + (2*n + n^2 + 178))*sizeof(Float64))
    gfree = CuDynamicSharedArray(Float64, n, (12*n + (2*n + n^2 + 178))*sizeof(Float64))
    dsave = CuDynamicSharedArray(Float64, n, (13*n + (2*n + n^2 + 178))*sizeof(Float64))
    indfree = CuDynamicSharedArray(Int, n, (14*n + (2*n + n^2 + 178))*sizeof(Float64))
    iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (14*n + (2*n + n^2 + 178))*sizeof(Float64))
    isave = CuDynamicSharedArray(Int, n, (3*n)*sizeof(Int) + (14*n + (2*n + n^2 + 178))*sizeof(Float64))


    A = CuDynamicSharedArray(Float64, (n,n), (14*n + (2*n + n^2 + 178))*sizeof(Float64)+(4*n)*sizeof(Int))
    B = CuDynamicSharedArray(Float64, (n,n), (14*n+n^2 + (2*n + n^2 + 178))*sizeof(Float64)+(4*n)*sizeof(Int))
    L = CuDynamicSharedArray(Float64, (n,n), (14*n+2*n^2 + (2*n + n^2 + 178))*sizeof(Float64)+(4*n)*sizeof(Int))

    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    CUDA.sync_threads()

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

    while search

        if task == 0 || task == 1

            f = eval_f_kernel(x,H,b)
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_g_kernel(x,g,H,b)
            eval_h_kernel(A,H)
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = ExaAdmm.ExaTron.dnrm2(n, g, 1)
            delta = gnorm0 
        end

        if search
            delta, task = ExaAdmm.ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                        isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
        end

         # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaAdmm.ExaTron.dgpnorm(n, x, xl, xu, g)

            if gnorm_inf <= gtol
                task = 4
            end

            if minor_iter >= max_minor
                status = 1
                search = false
            end
        end

        # [4] CONV: convergence was achieved.
        # [10] : warning fval is less than fmin

        if task == 4 || task == 10
            search = false
        end
     end # end while 
    
     
    CUDA.sync_threads()

    return status, minor_iter
end

@inline function eval_f_kernel(x::CuDeviceArray{Float64,1},A::CuDeviceArray{Float64,2},b::CuDeviceArray{Float64,1}) #f gpu
    f = 0.0
    tx = threadIdx().x
        @inbounds begin
            for i = 1:6
                for j = 1:6
                    f += 0.5*x[i]*A[i,j]*x[j]
                end
                    f += b[i]*x[i]
            end
        end
    CUDA.sync_threads()
    return f 
end

@inline function eval_g_kernel(x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2}, b::CuDeviceArray{Float64,1})
    tx = threadIdx().x
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
    end
    CUDA.sync_threads()
end

@inline function eval_h_kernel(A::CuDeviceArray{Float64,2}, H::CuDeviceArray{Float64,2})
    tx = threadIdx().x
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