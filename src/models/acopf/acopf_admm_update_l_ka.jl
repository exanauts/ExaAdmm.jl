@kernel function update_l_kernel_ka(
    n::Int, l, z,
    lz, beta::Float64
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        @inbounds begin
            l[tx] = -(lz[tx] + beta*z[tx])
        end
    end
end

function admm_update_l(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device
)
    par, sol, info = env.params, mod.solution, mod.info
    nblk_nvar = div(mod.nvar-1, 64)+1
    ev = update_l_kernel_ka(device,64,64*nblk_nvar)(
        mod.nvar, sol.l_curr, sol.z_curr, sol.lz, par.beta
    )
    KA.synchronize(device)
    info.time_l_update += 0.0
    return
end
