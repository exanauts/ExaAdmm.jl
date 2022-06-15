"""
    admm_outer_prestep()
    
at start of each outer loop, preset info.norm_z_prev
"""

function admm_outer_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = norm(sol.z_curr)
    return
end





"""
    admm_inner_prestep()
    
at start of each inner loop, preset sol.z_prev
"""

function admm_inner_prestep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    sol = mod.solution
    sol.z_prev .= sol.z_curr
    return
end





"""
    admm_poststep()
    
- after admm termination, fix solution pf_projection() and record time mod.info.time_projection
- update info.objval with projected solution
"""

function admm_poststep(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    data, sol, info, grid_data = env.data, mod.solution, mod.info, mod.grid_data

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    info.objval = sum(grid_data.c2[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)])^2 +
                        grid_data.c1[g]*(grid_data.baseMVA*sol.u_curr[mod.gen_start+2*(g-1)]) +
                        grid_data.c0[g] for g in 1:grid_data.ngen) + 
                    sum(0.5*dot(mod.sqp_line[:,l],mod.Hs[6*(l-1)+1:6*l,1:6],mod.sqp_line[:,l]) for l=1:grid_data.nline) 
end