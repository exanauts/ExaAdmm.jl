"""
Driver to run TRON on GPU. This should be called from a kernel.
"""
@inline function ucmp_tron_generator_kernel(
    t::Int, n::Int, max_feval::Int, max_minor::Int, gtol::Float64, scale::Float64,
    x::CuDeviceArray{Float64,1}, xl::CuDeviceArray{Float64,1}, xu::CuDeviceArray{Float64,1},
    param::CuDeviceArray{Float64,2},
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)

    tx = threadIdx().x
    I = blockIdx().x

    g = CuDynamicSharedArray(Float64, n, (3*n+18)*sizeof(Float64))
    xc = CuDynamicSharedArray(Float64, n, (4*n+18)*sizeof(Float64))
    s = CuDynamicSharedArray(Float64, n, (5*n+18)*sizeof(Float64))
    wa = CuDynamicSharedArray(Float64, n, (6*n+18)*sizeof(Float64))
    wa1 = CuDynamicSharedArray(Float64, n, (7*n+18)*sizeof(Float64))
    wa2 = CuDynamicSharedArray(Float64, n, (8*n+18)*sizeof(Float64))
    wa3 = CuDynamicSharedArray(Float64, n, (9*n+18)*sizeof(Float64))
    wa4 = CuDynamicSharedArray(Float64, n, (10*n+18)*sizeof(Float64))
    wa5 = CuDynamicSharedArray(Float64, n, (11*n+18)*sizeof(Float64))
    gfree = CuDynamicSharedArray(Float64, n, (12*n+18)*sizeof(Float64))
    dsave = CuDynamicSharedArray(Float64, n, (13*n+18)*sizeof(Float64))
    indfree = CuDynamicSharedArray(Int, n, (14*n+18)*sizeof(Float64))
    iwa = CuDynamicSharedArray(Int, 2*n, n*sizeof(Int) + (14*n+18)*sizeof(Float64))
    isave = CuDynamicSharedArray(Int, n, (3*n)*sizeof(Int) + (14*n+18)*sizeof(Float64))

    A = CuDynamicSharedArray(Float64, (n,n), (14*n+18)*sizeof(Float64)+(4*n)*sizeof(Int))
    B = CuDynamicSharedArray(Float64, (n,n), (14*n+18+n^2)*sizeof(Float64)+(4*n)*sizeof(Int))
    L = CuDynamicSharedArray(Float64, (n,n), (14*n+18+2*n^2)*sizeof(Float64)+(4*n)*sizeof(Int))

    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    CUDA.sync_threads()

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

        # [0|1]: Evaluate function.

        if task == 0 || task == 1
            f = eval_f_generator_continuous_kernel_gpu(t, I, x, param, scale, c2, c1, c0, baseMVA)
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end
        

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_grad_f_generator_continuous_kernel_gpu(t, I, x, g, param, scale, c2, c1, c0, baseMVA)
            eval_h_generator_continuous_kernel_gpu(t, I, x, A, param, scale, c2, c1, c0, baseMVA)
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = ExaTron.dnrm2(n, g, 1)
            delta = gnorm0
        end

        # Call Tron.

        if search
            delta, task = ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                        isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaTron.dgpnorm(n, x, xl, xu, g)

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
        #=
        =#
    end

    CUDA.sync_threads()

    return status, minor_iter
end
