function mpacopf_compute_primal_residual_kernel(
    n::Int, gen_start::Int,
    rp::CuDeviceArray{Float64,1}, u::CuDeviceArray{Float64,1},
    v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1}
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= n
        gid = gen_start + 2*(tx-1)
        rp[tx] = u[tx] - v[gid] + z[tx]
    end
    return
end

function admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    info, par = mod.info, env.params

    info.primres = 0.0
    info.dualres = 0.0
    info.primsca = 0.0
    info.dualsca = 0.0
    info.norm_z_curr = 0.0
    info.mismatch = 0.0

    for i=1:mod.len_horizon
        admm_update_residual(env, mod.models[i], normalized = false)
        info.primsca = max(info.primsca, mod.models[i].info.primsca)
        info.dualsca = max(info.dualsca, mod.models[i].info.dualsca)
    #=
        info.primres += mod.models[i].info.primres^2
        info.dualres += mod.models[i].info.dualres^2
        info.norm_z_curr += mod.models[i].info.norm_z_curr^2
        info.mismatch += mod.models[i].info.mismatch^2
    =#
    end

    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i-1], mod.solution[i]
        @cuda threads=64 blocks=(div(submod.grid_data.ngen-1, 64)+1) mpacopf_compute_primal_residual_kernel(submod.grid_data.ngen, submod.gen_start, sol_ramp.rp, sol_ramp.u_curr, submod.solution.v_curr, sol_ramp.z_curr)
        @cuda threads=64 blocks=(div(submod.grid_data.ngen-1, 64)+1) vector_difference(submod.grid_data.ngen, sol_ramp.rd, sol_ramp.z_curr, sol_ramp.z_prev)
        CUDA.synchronize()
        @cuda threads=64 blocks=(div(submod.grid_data.ngen-1, 64)+1) vector_difference(submod.grid_data.ngen, sol_ramp.Ax_plus_By, sol_ramp.rp, sol_ramp.z_curr)

        mod.models[i].info.primres = sqrt(mod.models[i].info.primres^2 + CUDA.norm(sol_ramp.rp)^2)
        mod.models[i].info.dualres = sqrt(mod.models[i].info.dualres^2 + CUDA.norm(sol_ramp.rd)^2)
        mod.models[i].info.norm_z_curr = sqrt(mod.models[i].info.norm_z_curr^2 + CUDA.norm(sol_ramp.z_curr)^2)
        mod.models[i].info.mismatch = sqrt(mod.models[i].info.mismatch^2 + CUDA.norm(sol_ramp.Ax_plus_By)^2)
        info.primsca = max(info.primsca, CUDA.norm(sol_ramp.u_curr), CUDA.norm(sol_ramp.v_curr), CUDA.norm(sol_ramp.z_curr))
        info.dualsca = max(info.dualsca, CUDA.norm(sol_ramp.l_curr))
    #=
        info.primres += norm(sol_ramp.rp)^2
        info.dualres += norm(sol_ramp.rd)^2
        info.norm_z_curr += norm(sol_ramp.z_curr)^2
        info.mismatch += norm(sol_ramp.Ax_plus_By)^2
    =#
    end

    for i=1:mod.len_horizon
        info.primres = max(info.primres, mod.models[i].info.primres)
        info.dualres = max(info.dualres, mod.models[i].info.dualres)
        info.norm_z_curr = max(info.norm_z_curr, mod.models[i].info.norm_z_curr)
        info.mismatch = max(info.mismatch, mod.models[i].info.mismatch)
    end
    info.primres /= info.primsca
    info.dualres /= info.dualsca
    info.primtol = sqrt(mod.nvar) * par.ABSTOL / info.primsca + par.RELTOL
    info.dualtol = sqrt(mod.nvar) * par.ABSTOL / info.dualsca + par.RELTOL

    #=
    info.primres = sqrt(info.primres)
    info.dualres = sqrt(info.dualres)
    info.norm_z_curr = sqrt(info.norm_z_curr)
    info.mismatch = sqrt(info.mismatch)
    =#
    return
end