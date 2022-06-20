function admm_outer_prestep(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device::KA.GPU
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = norm(sol.z_curr)
    return
end

function admm_inner_prestep(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    device::KA.GPU
)
    sol = mod.solution
    wait(copy_data_kernel_ka(device, 64, mod.nvar)(
            mod.nvar, sol.z_prev, sol.z_curr,
            dependencies=Event(device)
        )
    )
    return
end

function admm_poststep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::AbstractOPFModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::KA.GPU
)
    data, sol, info, grid_data = env.data, mod.solution, mod.info, mod.grid_data

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    u_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)

    info.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(grid_data.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                      data.generators[g].coeff[data.generators[g].n-1]*(grid_data.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                      data.generators[g].coeff[data.generators[g].n]
                      for g in 1:grid_data.ngen)::Float64
end
