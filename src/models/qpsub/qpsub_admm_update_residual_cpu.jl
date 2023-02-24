"""
    admm_update_residual()

- compute termination errors and other info
- update info.primres, info.dualres, info.mismatch, info.objval, info.auglag
- update sol.rp, sol.rd, sol.Ax_plus_By
- only used in one-level ADMM
"""

function admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol, info, data, par, grid_data = mod.solution, mod.info, env.data, env.params, mod.grid_data


    sol.rp .= sol.u_curr .- sol.v_curr #u-v
    sol.rd .= sol.rho .* (sol.v_curr - mod.v_prev)#from Boyd's single-level admm
    sol.Ax_plus_By .= sol.rp #x-xbar

    info.primres = norm(sol.rp)
    info.dualres = norm(sol.rd)
    info.mismatch = norm(sol.Ax_plus_By)
    

    info.objval = sum(mod.qpsub_c2[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                        mod.qpsub_c1[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])
                        for g in 1:grid_data.ngen) + 
                            sum(0.5*dot(mod.sqp_line[:,l],mod.Hs[6*(l-1)+1:6*l,1:6],mod.sqp_line[:,l]) for l=1:grid_data.nline) 
    
    info.auglag = info.objval + sum(sol.lz[i]*sol.z_curr[i] for i=1:mod.nvar) +
                  0.5*par.beta*sum(sol.z_curr[i]^2 for i=1:mod.nvar) +
                  sum(sol.l_curr[i]*sol.rp[i] for i=1:mod.nvar) +
                  0.5*sum(sol.rho[i]*(sol.rp[i])^2 for i=1:mod.nvar)
    


    return
end