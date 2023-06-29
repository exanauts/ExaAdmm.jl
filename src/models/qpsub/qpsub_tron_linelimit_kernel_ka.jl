@inline function tron_gpu_test(
    n, H, b, x, xl, xu,
    I, J
)
    tx = J


    g = @localmem Float64 (n,)
    xc = @localmem Float64 (n,)
    s = @localmem Float64 (n,)
    wa = @localmem Float64 (n,)
    wa1 = @localmem Float64 (n,)
    wa2 = @localmem Float64 (n,)
    wa3 = @localmem Float64 (n,)
    wa4 = @localmem Float64 (n,)
    wa5 = @localmem Float64 (n,)
    gfree = @localmem Float64 (n,)
    dsave = @localmem Float64 (n,)
    indfree = @localmem Int (n,)
    iwa = @localmem Int (2*n,)
    isave = @localmem Int (n,)


    A = @localmem Float64 (n,n)
    B = @localmem Float64 (n,n)
    L = @localmem Float64 (n,n)

    if tx <= n
        @inbounds for j=1:n
            A[tx,j] = 0.0
            B[tx,j] = 0.0
            L[tx,j] = 0.0
        end
    end
    @synchronize

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

            f = eval_f_kernel(x,H,b, I, J)
            nfev += 1
            if nfev >= max_feval
                search = false
            end
        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            eval_g_kernel(x,g,H,b, I, J)
            eval_h_kernel(A,H, I, J)
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = ExaAdmm.ExaTron.dnrm2(n, g, 1, J)
            delta = gnorm0
        end

        if search
            delta, task = ExaAdmm.ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                        isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, J)
        end

         # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaAdmm.ExaTron.dgpnorm(n, x, xl, xu, g, J)

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


    @synchronize

    return status, minor_iter
end

@inline function eval_f_kernel(x,A,b,I,J) #f gpu
    f = 0.0
        @inbounds begin
            for i = 1:6
                for j = 1:6
                    f += 0.5*x[i]*A[i,j]*x[j]
                end
                    f += b[i]*x[i]
            end
        end
    @synchronize
    return f
end

@inline function eval_g_kernel(x, g, A, b, I, J)
    tx = J
    g1 = A[1,1]*x[1] + A[1,2]*x[2] + A[1,3]*x[3] + A[1,4]*x[4] + A[1,5]*x[5] + A[1,6]*x[6] + b[1]
    g2 = A[2,1]*x[1] + A[2,2]*x[2] + A[2,3]*x[3] + A[2,4]*x[4] + A[2,5]*x[5] + A[2,6]*x[6] + b[2]
    g3 = A[3,1]*x[1] + A[3,2]*x[2] + A[3,3]*x[3] + A[3,4]*x[4] + A[3,5]*x[5] + A[3,6]*x[6] + b[3]
    g4 = A[4,1]*x[1] + A[4,2]*x[2] + A[4,3]*x[3] + A[4,4]*x[4] + A[4,5]*x[5] + A[4,6]*x[6] + b[4]
    g5 = A[5,1]*x[1] + A[5,2]*x[2] + A[5,3]*x[3] + A[5,4]*x[4] + A[5,5]*x[5] + A[5,6]*x[6] + b[5]
    g6 = A[6,1]*x[1] + A[6,2]*x[2] + A[6,3]*x[3] + A[6,4]*x[4] + A[6,5]*x[5] + A[6,6]*x[6] + b[6]
    if tx == 1
        g[1] = g1
        g[2] = g2
        g[3] = g3
        g[4] = g4
        g[5] = g5
        g[6] = g6
    end
    CUDA.sync_threads()
end

@inline function eval_h_kernel(A, H, I, J)
    tx = J
    if tx == 1
        @inbounds begin
            for i = 1:6
                for j = 1:6
                    A[i,j] = H[i,j]
                end
            end
        end
    end
    @synchronize
end
