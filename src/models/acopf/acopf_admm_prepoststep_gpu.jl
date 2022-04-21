function admm_outer_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelAcopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol, info = mod.solution, mod.info
    info.norm_z_prev = CUDA.norm(sol.z_curr)
    return
end

function admm_inner_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelAcopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    sol = mod.solution
    @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.z_prev, sol.z_curr)
    #@cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.rp_prev, sol.rp)
    CUDA.synchronize()
    return
end

function admm_poststep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelAcopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    data, sol, info = env.data, mod.solution, mod.info

    if env.use_projection
        time_projection = @timed pf_projection(env, mod)
        mod.info.time_projection = time_projection.time
    end

    u_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)

    info.objval = sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)])^2 +
                      data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*(g-1)]) +
                      data.generators[g].coeff[data.generators[g].n]
                      for g in 1:mod.ngen)::Float64
end