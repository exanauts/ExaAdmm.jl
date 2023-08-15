function compute_primal_residual_kernel(n::Int, rp::CuDeviceArray{Float64,1}, u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = u[tx] - v[tx] + z[tx]
    end

    return
end

function admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::AbstractOPFModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing;
    normalized=true
)
    sol, info, par = mod.solution, mod.info, env.params

    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_kernel(mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr)
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.rd, sol.z_curr, sol.z_prev)
    CUDA.synchronize()
    CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr)

    info.primsca = max(CUDA.norm(sol.u_curr), CUDA.norm(sol.v_curr), CUDA.norm(sol.z_curr))
    info.dualsca = CUDA.norm(sol.l_curr)
    info.primres = CUDA.norm(sol.rp)
    info.dualres = CUDA.norm(sol.rd)
    info.primtol = sqrt(mod.nvar) * par.ABSTOL + par.RELTOL * info.primsca
    info.dualtol = sqrt(mod.nvar) * par.ABSTOL + par.RELTOL * info.dualsca
    if normalized
        info.primres /= info.primsca
        info.dualres /= info.dualsca
        info.primtol /= info.primsca
        info.dualtol /= info.dualsca
    end
    info.norm_z_curr = CUDA.norm(sol.z_curr)
    info.mismatch = CUDA.norm(sol.Ax_plus_By)

    return
end
