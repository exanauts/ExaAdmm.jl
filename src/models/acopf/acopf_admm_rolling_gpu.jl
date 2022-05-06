function update_real_power_current_bounds(
    ngen::Int, gen_start::Int,
    pgmin_curr::CuDeviceArray{Float64,1}, pgmax_curr::CuDeviceArray{Float64,1},
    pgmin_orig::CuDeviceArray{Float64,1}, pgmax_orig::CuDeviceArray{Float64,1},
    ramp_rate::CuDeviceArray{Float64,1}, x_curr::CuDeviceArray{Float64,1})

    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= ngen
        pg_idx = gen_start + 2*(g-1)
        pgmin_curr[g] = max(pgmin_orig[g], x_curr[pg_idx] - ramp_rate[g])
        pgmax_curr[g] = min(pgmax_orig[g], x_curr[pg_idx] + ramp_rate[g])
    end
    return
end

function admm_restart_rolling(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::Model{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}};
    start_period=1, end_period=6, result_file="warm-start")

    @assert env.load_specified == true
    @assert start_period >= 1 && end_period <= size(env.load.pd,2)

    nblk_gen = div(mod.ngen-1, 64) + 1

    io = open(result_file*"_tight-factor"*string(env.tight_factor)*".txt", "w")
    @printf(io, " ** Parameters\n")
    @printf(io, "tight_factor    = %.6e\n", env.tight_factor)
    @printf(io, "rho_pq          = %.6e\n", env.initial_rho_pq)
    @printf(io, "rho_va          = %.6e\n", env.initial_rho_va)
    @printf(io, "scale           = %.6e\n", env.params.scale)
    @printf(io, "outer_iterlim   = %5d\n", env.params.outer_iterlim)
    @printf(io, "inner_iterlim   = %5d\n", env.params.inner_iterlim)
    @printf(io, "outer_eps       = %.6e\n", env.params.outer_eps)
    flush(io)

    filename = result_file*"_tight-factor"*string(env.tight_factor)*"_solution"
    u_curr = zeros(length(mod.solution.u_curr))
    v_curr = zeros(length(mod.solution.v_curr))
    for t=start_period:end_period
        mod.Pd = env.load.pd[:,t]
        mod.Qd = env.load.qd[:,t]
        admm_restart(env, mod)

        # Uncomment the following three lines if you want to save the solution.
        #copyto!(u_curr, mod.solution.u_curr)
        #copyto!(v_curr, mod.solution.v_curr)
        #save(filename*"_t"*string(t)*".jld2", "u_curr", u_curr, "v_curr", v_curr)

        @printf(" ** Statistics of time period %d\n", t)
        @printf("Status  . . . . . . . . . . . . . . . . . %s\n", mod.info.status)
        @printf("Objective value . . . . . . . . . . . . . %.6e\n", mod.info.objval)
        @printf("Residual  . . . . . . . . . . . . . . . . %.6e\n", mod.info.mismatch)
        @printf("Cumulative iterations . . . . . . . . . . %5d\n", mod.info.cumul)
        #@printf("Constraint violations (except line) . . . %.6e\n", mod.solution.max_viol_except_line)
        #@printf("Line violations (RateA) . . . . . . . . . %.6e\n", mod.solution.max_line_viol_rateA)
        @printf("Time (secs) . . . . . . . . . . . . . . . %5.3f\n", mod.info.time_overall + mod.info.time_projection)

        @printf(io, " ** Statistics of time period %d\n", t)
        @printf(io, "Status  . . . . . . . . . . . . . . . . . %s\n", mod.info.status)
        @printf(io, "Objective value . . . . . . . . . . . . . %.6e\n", mod.info.objval)
        @printf(io, "Residual  . . . . . . . . . . . . . . . . %.6e\n", mod.info.mismatch)
        @printf(io, "Cumulative iterations . . . . . . . . . . %5d\n", mod.info.cumul)
        #@printf(io, "Constraint violations (except line) . . . %.6e\n", mod.solution.max_viol_except_line)
        #@printf(io, "Line violations (RateA) . . . . . . . . . %.6e\n", mod.solution.max_line_viol_rateA)
        @printf(io, "Time (secs) . . . . . . . . . . . . . . . %5.3f\n", mod.info.time_overall + mod.info.time_projection)

        CUDA.@sync @cuda threads=64 blocks=nblk_gen update_real_power_current_bounds(mod.ngen, mod.gen_start,
            mod.pgmin_curr, mod.pgmax_curr, mod.pgmin, mod.pgmax,
            mod.ramp_rate, mod.solution.u_curr)

        flush(io)
    end

    close(io)
end