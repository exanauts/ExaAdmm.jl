"""
Driver to run TRON on GPU. This should be called from a kernel.
"""
@inline function tron_linelimit_kernel(
    n, shift::Int, max_feval::Int, max_minor::Int, gtol::Float64, scale::Float64, use_polar::Bool,
    x, xl, xu,param,
    YffR::Float64, YffI::Float64,
    YftR::Float64, YftI::Float64,
    YttR::Float64, YttI::Float64,
    YtfR::Float64, YtfI::Float64,
    I, J)

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
    wa = @localmem Float64 (n,)
    gfree = @localmem Float64 (n,)
    dsave = @localmem Float64 (n,)
    indfree = @localmem Int (n,)
    iwa = @localmem Int (n,)
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
#=
        if threadIdx().x == 1
            @cuprintf("iter = %d\n", minor_iter)
            for i=1:n
                @cuprintf("  x[%d] = %.16e\n", i, x[i])
            end
        end
=#
        # [0|1]: Evaluate function.

        if task == 0 || task == 1
            if use_polar
                f = eval_f_polar_linelimit_kernel(n, shift, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            else
                f = eval_f_kernel(n, scale, x, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            end
            nfev += 1
            if nfev >= max_feval
                search = false
            end

        end

        # [2] G or H: Evaluate gradient and Hessian.

        if task == 0 || task == 2
            if use_polar
                eval_grad_f_polar_linelimit_kernel(n, shift, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
                eval_h_polar_linelimit_kernel(n, shift, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            else
                eval_grad_f_kernel(n, scale, x, g, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
                eval_h_kernel(n, scale, x, A, param, YffR, YffI, YftR, YftI, YttR, YttI, YtfR, YtfI, I, J)
            end
            ngev += 1
            nhev += 1
            minor_iter += 1
        end

        # Initialize the trust region bound.

        if task == 0
            gnorm0 = ExaTron.dnrm2(n, g, 1, J)
            delta = gnorm0
        end

        # Call Tron.
#=
        if tx == 1
            @cuprintln("minor_iter = ", minor_iter, " task = ", task, " f = ", f)
        end
=#
        if search
            delta, task = ExaTron.dtron(n, x, xl, xu, f, g, A, frtol, fatol, fmin, cgtol,
                                        cg_itermax, delta, task, B, L, xc, s, indfree, gfree,
                                        isave, dsave, wa, iwa, wa1, wa2, wa3, wa4, wa5, J)
        end

        # [3] NEWX: a new point was computed.

        if task == 3
            gnorm_inf = ExaTron.dgpnorm(n, x, xl, xu, g, J)
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

        # [4] CONV: convergence was achieved.
        # [10] : warning fval is less than fmin

        if task == 4 || task == 10
            search = false
        end
    end

    @synchronize

    return status, minor_iter
end
