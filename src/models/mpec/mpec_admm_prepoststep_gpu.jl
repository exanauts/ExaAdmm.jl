function acopf_admm_outer_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = CUDA.norm(sol.z_curr)
    return
end

function acopf_admm_inner_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol = mod.solution
    sol.z_prev .= sol.z_curr
    return
end

function acopf_admm_poststep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    data, grid, sol, info = env.data, mod.grid, mod.solution, mod.info

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end
#=
    info.user.err_pg, info.user.err_qg = check_generator_bounds(mod, sol.u_curr)
    info.user.err_vm = check_voltage_bounds_alternative(mod, sol.v_curr)
    info.user.err_real, info.user.err_reactive = check_power_balance_alternative(mod, sol.u_curr, sol.v_curr)
    info.user.num_rateA_viols, info.user.err_rateA = check_linelimit_violation(mod, data, sol.v_curr)

    info.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(grid.baseMVA*sol.v_curr[mod.gen_start+2*(g-1)])^2 +
                      data.generators[g].coeff[data.generators[g].n-1]*(grid.baseMVA*sol.v_curr[mod.gen_start+2*(g-1)]) +
                      data.generators[g].coeff[data.generators[g].n]
                      for g in 1:grid.ngen)::Float64
=#
end