function acopf_admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    info::IterationInformation
)
    sol = mod.solution
    sol.rp .= sol.u_curr .- sol.v_curr .+ sol.z_curr
    sol.rd .= sol.z_curr .- sol.z_prev
    sol.Ax_plus_By .= sol.rp .- sol.z_curr

    info.primres = norm(sol.rp)
    info.dualres = norm(sol.rd)
    info.norm_z_curr = norm(sol.z_curr)
    info.mismatch = norm(sol.Ax_plus_By)

    return
end