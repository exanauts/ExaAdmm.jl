"""
Update variable `z`, representing the artificial variables that are driven to zero in the two-level ADMM.
"""
function admm_update_z(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    par, sol, info = env.params, mod.solution, mod.info
    ztime = @timed sol.z_curr .= (-(sol.lz .+ sol.l_curr .+ sol.rho.*(sol.u_curr .- sol.v_curr))) ./ (par.beta .+ sol.rho)
    info.time_z_update += ztime.time
    return
end
