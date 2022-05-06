function solve_pf(case::String; case_format="matpower", start_method="warm", use_gpu=false, tol=1e-6, max_iter=50)
    T = Float64; TD = Vector{Float64}
    pf = PowerFlow{T,TD}(case; case_format=case_format, start_method=start_method)
    solve_pf(pf; tol=tol, max_iter=max_iter)
end

function solve_pf(pf::PowerFlow, x::Vector{Float64}; tol=1e-6, max_iter=50)
    @assert length(x) == pf.n_var
    pf.x .= x
    solve_pf(pf; tol=tol, max_iter=max_iter)
end

function solve_pf(pf::PowerFlow; tol=1e-6, max_iter=50)
    nbus = length(pf.nw["bus"])
    pq_idx = findall(x -> Int(x["type"]) == 1, pf.nw["bus"])
    pv_idx = findall(x -> Int(x["type"]) == 2, pf.nw["bus"])
    slack_idx = findall(x -> Int(x["type"]) == 3, pf.nw["bus"])

    @printf("Bus statistics:\n")
    @printf("  # PQ    buses  = %5d\n", length(pq_idx))
    @printf("  # PV    buses  = %5d\n", length(pv_idx))
    @printf("  # Slack buses  = %5d\n", length(slack_idx))

    rslice = vcat([pf.equ_pg_start+(i-1) for i=1:length(pf.nw["bus"]) if Int(pf.nw["bus"][i]["type"]) != 3],
                  [pf.equ_qg_start+(i-1) for i in pq_idx])

    cslice = vcat([pf.var_vmva_start+2*(i-1)+1 for i in pv_idx],
                  [pf.var_vmva_start + k for i in pq_idx for k in (2*(i-1),2*(i-1)+1)])
    sort!(cslice)
    rx = view(pf.x, cslice)
    rF = view(pf.F, rslice)

    eval_f(pf, pf.x, pf.F)
    residual = maximum(abs.(pf.F[rslice]))

    @printf("\nSolving power flow using Newton-Raphson . . .\n")
    @printf("%12s    %12s\n", "Iteration", "Residual")
    @printf("%12d    %.6e\n", 0, residual)

    timed = @timed begin
        it = 0
        while it < max_iter && residual > tol
            it += 1
            eval_jac(pf, pf.x, pf.Jac)

            rJac = pf.Jac[rslice,cslice]
            dx = -(rJac\rF)
            rx .+= dx

            eval_f(pf, pf.x, pf.F)
            residual = maximum(abs.(rF))

            if (it % 50) == 0
                @printf("%12s    %12s\n", "Iteration", "Residual")
            end
            @printf("%12d    %.6e\n", it, residual)
        end
    end

    vm_idx = collect(pf.var_vmva_start:2:pf.var_vmva_start+2*(nbus-1))
    vmax, vmax_idx = findmax(pf.x[vm_idx])
    vmin, vmin_idx = findmin(pf.x[vm_idx])
    bndviol = max.(pf.xlo[vm_idx] .- pf.x[vm_idx], pf.x[vm_idx] .- pf.xup[vm_idx], 0.0)

    @printf("\n ** Results\n")
    @printf("Residual . . . . . . . . %.2e\n", residual)
    @printf("Vmax . . . . . . . . . . %8.2f (%d)\n", vmax, vmax_idx)
    @printf("Vmin . . . . . . . . . . %8.2f (%d)\n", vmin, vmin_idx)
    @printf("Max |v-vbnd| . . . . . . %.2e\n", maximum(bndviol))
    @printf("Elapsed time (secs). . . %8.2f\n", timed.time)

    return pf
end