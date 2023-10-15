"""
    admm_update_residual()

- compute termination errors and other info
- update info.primres, info.dualres, info.mismatch
- update sol.rp, sol.rd, sol.Ax_plus_By
- only used in one-level ADMM
- GPU kernel: compute_primal_residual_kernel_qpsub, compute_dual_residual_kernel_qpsub, copy_data_kernel
"""

@kernel function compute_primal_residual_kernel_qpsub_ka(n::Int, rp, u, v)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        rp[tx] = u[tx] - v[tx] #u-v or x-xbar
    end
end

@kernel function compute_dual_residual_kernel_qpsub_ka(n::Int, rd, v_curr, v_prev, rho)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        rd[tx] = rho[tx] * (v_curr[tx] - v_prev[tx]) #from Boyd's single-level admm
    end
end


function admm_update_residual(
    env::AdmmEnv,
    mod::ModelQpsub,
    device
)
    sol, info, data, par, grid_data = mod.solution, mod.info, env.data, env.params, mod.grid_data
    nblk_nvar = div(mod.nvar-1, 64)+1
    ev = compute_primal_residual_kernel_qpsub_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.rp, sol.u_curr, sol.v_curr
    )
    KA.synchronize(device)

    ev = compute_dual_residual_kernel_qpsub_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.rd, sol.v_curr, mod.v_prev, sol.rho
    )
    KA.synchronize(device)

    ev = copy_data_kernel_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.Ax_plus_By, sol.rp
    ) # from gpu utility
    KA.synchronize(device)

    info.primsca = max(norm(sol.u_curr, device), norm(sol.v_curr, device))
    info.dualsca = norm(sol.l_curr, device)
    info.primres = norm(sol.rp, device) / info.primsca
    info.dualres = norm(sol.rd, device) / info.dualsca
    info.primtol = sqrt(mod.nvar) * par.ABSTOL / info.primsca + par.RELTOL
    info.dualtol = sqrt(mod.nvar) * par.ABSTOL / info.dualsca + par.RELTOL
    info.mismatch = norm(sol.Ax_plus_By, device)

    return
end
