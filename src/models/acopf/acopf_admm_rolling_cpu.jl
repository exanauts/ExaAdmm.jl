function update_real_power_current_bounds(
    ngen::Int, gen_start::Int,
    pgmin_curr::Array{Float64,1}, pgmax_curr::Array{Float64,1},
    pgmin_orig::Array{Float64,1}, pgmax_orig::Array{Float64,1},
    ramp_rate::Array{Float64,1}, x_curr::Array{Float64,1})

    pg_idx = gen_start
    for g=1:ngen
        pgmin_curr[g] = max(pgmin_orig[g], x_curr[pg_idx] - ramp_rate[g])
        pgmax_curr[g] = min(pgmax_orig[g], x_curr[pg_idx] + ramp_rate[g])
        pg_idx += 2
    end
end

#function admm_restart_rolling(
#    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
#    mod::Model{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}};
#    start_period=1, end_period=6, update_start_period_bounds=false, result_file="warm-start.txt")

#    @assert env.load_specified == true
#    @assert start_period >= 1 && end_period <= size(env.load.pd,2)

#    par = env.params
#    sol = mod.solution

#    if update_start_period_bounds == true
#        update_real_power_current_bounds(mod.ngen, mod.gen_start,
#            mod.pgmin_curr, mod.pgmax_curr, mod.pgmin, mod.pgmax,
#            mod.ramp_rate, mod.solution.x_curr)
#    end

#    for t=start_period:end_period
#        mod.Pd .= env.load.pd[:,t]
#        mod.Qd .= env.load.qd[:,t]
#        admm_restart(env, mod)
#        @printf(" ** Statistics of time period %d\n", t)
#        @printf("Status  . . . . . . . . . . . . . . . . . %s\n", mod.info.status)
#        @printf("Objective value . . . . . . . . . . . . . %.6e\n", mod.info.objval)
#        @printf("Residual  . . . . . . . . . . . . . . . . %.6e\n", mod.info.mismatch)
#        #@printf("Constraint violations (except line) . . . %.6e\n", mod.solution.max_viol_except_line)
#        #@printf("Line violations (RateA) . . . . . . . . . %.6e\n", mod.solution.max_line_viol_rateA)
#        @printf("Time (secs) . . . . . . . . . . . . . . . %5.3f\n", mod.info.time_overall)
#        update_real_power_current_bounds(mod.ngen, mod.gen_start,
#            mod.pgmin_curr, mod.pgmax_curr, mod.pgmin, mod.pgmax,
#            mod.ramp_rate, mod.solution.u_curr)
#    end
#end
