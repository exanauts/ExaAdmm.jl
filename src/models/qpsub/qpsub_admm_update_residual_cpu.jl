"""
    acopf_admm_update_residual()

- compute termination errors and other info
- update info.primres, info.dualres, info.norm_z_curr, info.mismatch, info. objval
- update sol.rp, sol.rd, sol.Ax_plus_By
"""

function admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol, info, data, par, grid_data = mod.solution, mod.info, env.data, env.params, mod.grid_data

    sol.rp .= sol.u_curr .- sol.v_curr .+ sol.z_curr #x-xbar+z_curr
    sol.rd .= sol.z_curr .- sol.z_prev #? NOT USED
    sol.Ax_plus_By .= sol.rp .- sol.z_curr #x-xbar

    info.primres = norm(sol.rp)
    info.dualres = norm(sol.rd) #? NOT USED
    info.norm_z_curr = norm(sol.z_curr) #? NOT USED
    info.mismatch = norm(sol.Ax_plus_By)
    info.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                      data.generators[g].coeff[data.generators[g].n-1]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)]) +
                      data.generators[g].coeff[data.generators[g].n]
                      for g in 1:grid_data.ngen)::Float64
    info.auglag = info.objval + sum(sol.lz[i]*sol.z_curr[i] for i=1:length(mod.nvar)) +
                  0.5*par.beta*sum(sol.z_curr[i]^2 for i=1:length(mod.nvar)) +
                  sum(sol.l_curr[i]*sol.rp[i] for i=1:length(mod.nvar)) +
                  0.5*sum(sol.rho[i]*(sol.rp[i])^2 for i=1:length(mod.nvar))

    return
end