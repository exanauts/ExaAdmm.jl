"""
    admm_update_residual()

- compute termination errors and other info
- update info.primres, info.dualres, info.norm_z_curr, info.mismatch, info. objval
- update sol.rp, sol.rd, sol.Ax_plus_By
"""

function compute_primal_residual_kernel_qpsub(n::Int, rp::CuDeviceArray{Float64,1}, u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rp[tx] = u[tx] - v[tx]
    end

    return
end

function compute_dual_residual_kernel_qpsub(n::Int, rd::CuDeviceArray{Float64,1}, v_curr::CuDeviceArray{Float64,1}, v_prev::CuDeviceArray{Float64,1}, rho)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        rd[tx] = rho[tx] * (v_curr[tx] - v_prev[tx])
    end

    return
end


function admm_update_residual(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info, data, par, grid_data = mod.solution, mod.info, env.data, env.params, mod.grid_data

    #? two level
    # sol.rp .= sol.u_curr .- sol.v_curr .+ sol.z_curr #?x-xbar+z_curr 
    # sol.rd .= sol.z_curr .- sol.z_prev 
    # sol.Ax_plus_By .= sol.rp .- sol.z_curr #x-xbar
    # info.norm_z_curr = norm(sol.z_curr) #? NOT USED one level 

    #? one level (no z and new rd)
    sol.rp .= sol.u_curr .- sol.v_curr #? for debug
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_kernel_qpsub(mod.nvar, sol.rp, sol.u_curr, sol.v_curr)
    # sol.rd .= sol.rho .* (sol.v_curr - mod.v_prev) #? for debug single level admm from Boyd
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_dual_residual_kernel_qpsub(mod.nvar, sol.rd, sol.v_curr, mod.v_prev, sol.rho)
    # sol.Ax_plus_By .= sol.rp #? for debug x-xbar
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.Ax_plus_By, sol.rp) # from gpu utility

    # info.primres = norm(sol.rp)
    info.primres = CUDA.norm(sol.rp)
    # info.dualres = norm(sol.rd)
    info.dualres = CUDA.norm(sol.rd)
    # info.mismatch = norm(sol.Ax_plus_By)
    info.mismatch = CUDA.norm(sol.Ax_plus_By)
    

    #? not update until poststep following YD
    # info.objval = sum(mod.qpsub_c2[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
    #                     mod.qpsub_c1[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])
    #                     for g in 1:grid_data.ngen) + 
    #                         sum(0.5*dot(mod.sqp_line[:,l],mod.Hs[6*(l-1)+1:6*l,1:6],mod.sqp_line[:,l]) for l=1:grid_data.nline) 
    
    # info.auglag = info.objval + sum(sol.lz[i]*sol.z_curr[i] for i=1:mod.nvar) +
    #               0.5*par.beta*sum(sol.z_curr[i]^2 for i=1:mod.nvar) +
    #               sum(sol.l_curr[i]*sol.rp[i] for i=1:mod.nvar) +
    #               0.5*sum(sol.rho[i]*(sol.rp[i])^2 for i=1:mod.nvar)
    


    return
end