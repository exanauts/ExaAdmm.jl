@kernel function update_zv_kernel_ka(n::Int, u, v, z, l, rho, lz, beta)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        @inbounds begin
            z[tx] = (-(lz[tx] + l[tx] + rho[tx]*(u[tx] - v[tx]))) / (beta + rho[tx])
        end
    end
end

function admm_update_z(
    env,
    mod,
    device
)
    par, sol, info = env.params, mod.solution, mod.info

    update_zv_kernel_ka(device,64,mod.nvar)(
        mod.nvar, sol.u_curr, sol.v_curr, sol.z_curr,
        sol.l_curr, sol.rho, sol.lz, par.beta
    )
    KA.synchronize(device)
    info.time_z_update += 0.0
    return
end
