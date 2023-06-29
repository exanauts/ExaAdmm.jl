@kernel function update_lz_kernel_ka(n::Int, max_limit::Float64, z, lz, beta::Float64)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        @inbounds begin
            lz[tx] = max(-max_limit, min(max_limit, lz[tx] + beta*z[tx]))
        end
    end
end

function admm_update_lz(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device
)
    par, sol, info = env.params, mod.solution, mod.info
    nblk_nvar = div(mod.nvar-1, 64)+1
    update_lz_kernel_ka(device,64,64*nblk_nvar)(
        mod.nvar, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, par.beta
    )
    KA.synchronize(device)
    info.time_lz_update += 0.0
    return
end
