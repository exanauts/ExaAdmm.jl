"""
    admm_update_l()

- update l
- record time info.time_l_update
- only used in one-level ADMM
"""

function admm_update_l_single(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ModelQpsub{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    par, sol, info = env.params, mod.solution, mod.info
    sol.l_prev = sol.l_curr
    ltime = @timed sol.l_curr .= sol.l_prev + sol.rho .* (sol.u_curr-sol.v_curr)
    info.time_l_update += ltime.time
    return
end
