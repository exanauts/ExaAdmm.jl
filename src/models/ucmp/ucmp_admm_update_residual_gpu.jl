"""
Compute and update the primal and dual residuals at current iteration.

### Arguments

- `env::AdmmEnv` -- struct that defines the environment of ADMM
- `mod::UCMPModel` -- struct that defines model

### Notes

The primal and dual residuals are stored in `mod.solution.rp` and `mod.solution.rd`, respectively.
"""
function ucmp_update_rp_rd_kernel(n::Int, T::Int, 
    rp::CuDeviceArray{Float64,2}, rd::CuDeviceArray{Float64,2}, 
    u::CuDeviceArray{Float64,2}, v::CuDeviceArray{Float64,2}, z::CuDeviceArray{Float64,2}, z_prev::CuDeviceArray{Float64,2}
)
    tx = threadIdx().x
    bx = blockIdx().x
    if bx <= n && tx <= T
        @inbounds begin
            rp[bx,tx] = u[bx,tx] - v[bx,tx] + z[bx,tx]
            rd[bx,tx] = z[bx,tx] - z_prev[bx,tx]
        end
    end
    return
end

function ucmp_update_Ax_plus_By_kernel(n::Int, T::Int, 
    Ax_plus_By::CuDeviceArray{Float64,2}, rp::CuDeviceArray{Float64,2}, z::CuDeviceArray{Float64,2}
)
    tx = threadIdx().x
    bx = blockIdx().x
    if bx <= n && tx <= T
        @inbounds begin
            Ax_plus_By[bx,tx] = rp[bx,tx] - z[bx,tx]
        end
    end
    return
end

function ucmp_update_vr_rp_rd_kernel(n::Int, t::Int,
    rp::CuDeviceArray{Float64,1}, rd::CuDeviceArray{Float64,1},
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,2}, z::CuDeviceArray{Float64,1}, z_prev::CuDeviceArray{Float64,1}
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= n
        @inbounds begin
            rp[tx] = u[tx] - v[tx,3*t-5] + z[tx]
            rd[tx] = z[tx] - z_prev[tx]
        end
    end
    return
end


function admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    info = mod.info
    ngen = mod.mpmodel.models[1].grid_data.ngen

    info.primres = 0.0
    info.dualres = 0.0
    info.norm_z_curr = 0.0
    info.mismatch = 0.0

    admm_update_residual(env, mod.mpmodel, device)

    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution

    # update residuals from uc solution, these residuals will be added to each model at time t
    nthreads = 2^ceil(Int, log2(3*mod.mpmodel.len_horizon))
    CUDA.@sync @cuda threads=nthreads blocks=ngen ucmp_update_rp_rd_kernel(ngen, 3*mod.mpmodel.len_horizon, uc_sol.rp, uc_sol.rd, uc_sol.u_curr, uc_sol.v_curr, uc_sol.z_curr, uc_sol.z_prev)
    CUDA.@sync @cuda threads=nthreads blocks=ngen ucmp_update_Ax_plus_By_kernel(ngen, 3*mod.mpmodel.len_horizon, uc_sol.Ax_plus_By, uc_sol.rp, uc_sol.z_curr)

    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        model.info.primres = sqrt(model.info.primres^2 + CUDA.norm(uc_sol.rp[:, 3*i-2:3*i])^2)
        model.info.dualres = sqrt(model.info.dualres^2 + CUDA.norm(uc_sol.rd[:, 3*i-2:3*i])^2)
        model.info.norm_z_curr = sqrt(model.info.norm_z_curr^2 + CUDA.norm(uc_sol.z_curr[:, 3*i-2:3*i])^2)
        model.info.mismatch = sqrt(model.info.mismatch^2 + CUDA.norm(uc_sol.Ax_plus_By[:, 3*i-2:3*i])^2)
    end

    # update residuals from v ramp solution, these residuals will be added to each model at time t
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        CUDA.@sync @cuda threads=64 blocks=(div(ngen-1,64)+1) ucmp_update_vr_rp_rd_kernel(ngen, i, vr_sol.rp, vr_sol.rd, vr_sol.u_curr, uc_sol.v_curr, vr_sol.z_curr, vr_sol.z_prev)
        CUDA.@sync @cuda threads=64 blocks=(div(ngen-1,64)+1) vector_difference(ngen, vr_sol.Ax_plus_By, vr_sol.rp, vr_sol.z_curr)

        model = mod.mpmodel.models[i]
        model.info.primres = sqrt(model.info.primres^2 + CUDA.norm(vr_sol.rp)^2)
        model.info.dualres = sqrt(model.info.dualres^2 + CUDA.norm(vr_sol.rd)^2)
        model.info.norm_z_curr = sqrt(model.info.norm_z_curr^2 + CUDA.norm(vr_sol.z_curr)^2)
        model.info.mismatch = sqrt(model.info.mismatch^2 + CUDA.norm(vr_sol.Ax_plus_By)^2)
    end

    for i=1:mod.mpmodel.len_horizon
        info.primres = max(info.primres, mod.mpmodel.models[i].info.primres)
        info.dualres = max(info.dualres, mod.mpmodel.models[i].info.dualres)
        info.norm_z_curr = max(info.norm_z_curr, mod.mpmodel.models[i].info.norm_z_curr)
        info.mismatch = max(info.mismatch, mod.mpmodel.models[i].info.mismatch)
    end

    info.objval = mod.mpmodel.info.objval
    info.auglag = mod.mpmodel.info.auglag
    # on_cost, off_cost = mod.uc_params.con, mod.uc_params.coff
    # add uc objective terms
    # for i=1:mod.mpmodel.len_horizon
    #     info.objval += sum(uc_sol.u_curr[:, 3*i-1] .* on_cost) + sum(uc_sol.u_curr[:, 3*i] .* off_cost)
    # end
    # TODO: update info.auglag with augmented terms from uc_solution and vr_solution?

    return
end
