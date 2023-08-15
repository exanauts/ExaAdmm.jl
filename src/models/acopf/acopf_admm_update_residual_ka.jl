@kernel function compute_primal_residual_kernel_ka(n::Int, rp, u, v, z)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        @inbounds begin
            rp[tx] = u[tx] - v[tx] + z[tx]
        end
    end
end

function admm_update_residual(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device;
    normalized=true
)
    sol, info, par = mod.solution, mod.info, env.params
    nblk_nvar = div(mod.nvar-1, 64)+1
    compute_primal_residual_kernel_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr
    )
    KA.synchronize(device)
    vector_difference_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.rd, sol.z_curr, sol.z_prev
    )
    KA.synchronize(device)
    vector_difference_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr
    )
    KA.synchronize(device)

    info.primsca = max(norm(sol.u_curr, device), norm(sol.v_curr, device), norm(sol.z_curr, device))
    info.dualsca = norm(sol.l_curr, device)
    info.primres = norm(sol.rp, device)
    info.dualres = norm(sol.rd, device)
    info.primtol = sqrt(mod.nvar) * par.ABSTOL + par.RELTOL * info.primsca
    info.dualtol = sqrt(mod.nvar) * par.ABSTOL + par.RELTOL * info.dualsca
    if normalized
        info.primres /= info.primsca
        info.dualres /= info.dualsca
        info.primtol /= info.primsca
        info.dualtol /= info.dualsca
    end
    info.norm_z_curr = norm(sol.z_curr, device)
    info.mismatch = norm(sol.Ax_plus_By, device)

    return
end
