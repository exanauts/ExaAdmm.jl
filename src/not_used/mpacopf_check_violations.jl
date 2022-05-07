function check_ramp_violations(mod::Model, u_curr::Vector{Float64}, u_prev::Vector{Float64}, ramp_rate::Vector{Float64})
    max_viol = 0.0
    for g=1:mod.ngen
        pg_idx = mod.gen_start + 2*(g-1)
        max_viol = max(max_viol, max(0.0, -(ramp_rate[g]-abs(u_curr[pg_idx]-u_prev[pg_idx]))))
    end
    return max_viol
end