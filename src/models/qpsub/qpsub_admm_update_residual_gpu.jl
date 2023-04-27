"""
    admm_update_residual()

- compute termination errors and other info
- update info.primres, info.dualres, info.mismatch
- update sol.rp, sol.rd, sol.Ax_plus_By
- only used in one-level ADMM
- GPU kernel: compute_primal_residual_kernel_qpsub, compute_dual_residual_kernel_qpsub, copy_data_kernel
"""

function compute_primal_residual_kernel_qpsub(n::Int, rp::CuDeviceArray{Float64,1}, u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = u[tx] - v[tx] #u-v or x-xbar
    end

    return
end

function compute_dual_residual_kernel_qpsub(n::Int, rd::CuDeviceArray{Float64,1}, v_curr::CuDeviceArray{Float64,1}, v_prev::CuDeviceArray{Float64,1}, rho)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rd[tx] = rho[tx] * (v_curr[tx] - v_prev[tx]) #from Boyd's single-level admm
    end

    return
end


function admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info, data, par, grid_data = mod.solution, mod.info, env.data, env.params, mod.grid_data

    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_kernel_qpsub(mod.nvar, sol.rp, sol.u_curr, sol.v_curr)

    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_dual_residual_kernel_qpsub(mod.nvar, sol.rd, sol.v_curr, mod.v_prev, sol.rho)

    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.Ax_plus_By, sol.rp) # from gpu utility


    info.primres = CUDA.norm(sol.rp)

    info.dualres = CUDA.norm(sol.rd)

    info.mismatch = CUDA.norm(sol.Ax_plus_By)

    return
end
