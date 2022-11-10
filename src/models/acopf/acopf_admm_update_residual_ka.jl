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
    device
)
    sol, info = mod.solution, mod.info

    ev = compute_primal_residual_kernel_ka(device,64,mod.nvar)(
        mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr,
        dependencies=Event(device)
    )
    wait(ev)
    ev = vector_difference_ka(device,64,mod.nvar)(
        mod.nvar, sol.rd, sol.z_curr, sol.z_prev,
        dependencies=Event(device)
    )
    wait(ev)
    ev = vector_difference_ka(device,64,mod.nvar)(
        mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr,
        dependencies=Event(device)
    )
    wait(ev)

    info.primres = norm(sol.rp)
    info.dualres = norm(sol.rd)
    info.norm_z_curr = norm(sol.z_curr)
    info.mismatch = norm(sol.Ax_plus_By)

    return
end
