"""
    admm_update_l()

- update l
- record time info.time_l_update
- only used in one-level ADMM
- GPU kernel: update_l_kernel_single
"""


@kernel function update_l_kernel_single_ka(
    n::Int, l_curr, l_prev, u,
    v, rho,
    )
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        @inbounds begin
            l_curr[tx] = l_prev[tx] + rho[tx] * (u[tx] - v[tx])
        end
    end
end



function admm_update_l_single(
    env::AdmmEnv,
    mod::ModelQpsub,
    device
    )
    par, sol, info = env.params, mod.solution, mod.info
    sol.l_prev = sol.l_curr
    nblk_nvar = div(mod.nvar-1, 64)+1
    ev = update_l_kernel_single_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.l_curr, sol.l_prev, sol.u_curr, sol.v_curr, sol.rho
    )
    KA.synchronize(device)
    info.time_l_update += 0.0
    return
end
