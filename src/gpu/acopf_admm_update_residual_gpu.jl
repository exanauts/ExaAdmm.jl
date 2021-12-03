function compute_primal_residual_kernel(n::Int, rp, u, v, z)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = u[tx] - v[tx] + z[tx]
    end

    return
end

function acopf_admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    info::IterationInformation
)
    sol = mod.solution

    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_kernel(mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr)
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.rd, sol.z_curr, sol.z_prev)
    CUDA.synchronize()
    CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr)

    info.primres = CUDA.norm(sol.rp)
    info.dualres = CUDA.norm(sol.rd)
    info.norm_z_curr = CUDA.norm(sol.z_curr)
    info.mismatch = CUDA.norm(sol.Ax_plus_By)

    return
end