function acopf_admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::MultiPeriodModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    info = mod.info

    info.primres = 0.0
    info.dualres = 0.0
    info.norm_z_curr = 0.0
    info.mismatch = 0.0

    for i=1:mod.len_horizon
        acopf_admm_update_residual(env, mod.models[i])
    #=
        info.primres += mod.models[i].info.primres^2
        info.dualres += mod.models[i].info.dualres^2
        info.norm_z_curr += mod.models[i].info.norm_z_curr^2
        info.mismatch += mod.models[i].info.mismatch^2
    =#
    end

    for i=2:mod.len_horizon
        submod, sol_ramp = mod.models[i-1], mod.solution[i]
        v_curr = @view submod.solution.v_curr[submod.gen_start:2:submod.gen_start+2*submod.ngen-1]
        sol_ramp.rp .= sol_ramp.u_curr .- v_curr .+ sol_ramp.z_curr
        sol_ramp.rd .= sol_ramp.z_curr .- sol_ramp.z_prev
        sol_ramp.Ax_plus_By .= sol_ramp.rp .- sol_ramp.z_curr

        mod.models[i].info.primres = sqrt(mod.models[i].info.primres^2 + norm(sol_ramp.rp)^2)
        mod.models[i].info.dualres = sqrt(mod.models[i].info.dualres^2 + norm(sol_ramp.rd)^2)
        mod.models[i].info.norm_z_curr = sqrt(mod.models[i].info.norm_z_curr^2 + norm(sol_ramp.z_curr)^2)
        mod.models[i].info.mismatch = sqrt(mod.models[i].info.mismatch^2 + norm(sol_ramp.Ax_plus_By)^2)
    #=
        info.primres += norm(sol_ramp.rp)^2
        info.dualres += norm(sol_ramp.rd)^2
        info.norm_z_curr += norm(sol_ramp.z_curr)^2
        info.mismatch += norm(sol_ramp.Ax_plus_By)^2
    =#
    end

    for i=1:mod.len_horizon
        info.primres = max(info.primres, mod.models[i].info.primres)
        info.dualres = max(info.dualres, mod.models[i].info.dualres)
        info.norm_z_curr = max(info.norm_z_curr, mod.models[i].info.norm_z_curr)
        info.mismatch = max(info.mismatch, mod.models[i].info.mismatch)
    end

    #=
    info.primres = sqrt(info.primres)
    info.dualres = sqrt(info.dualres)
    info.norm_z_curr = sqrt(info.norm_z_curr)
    info.mismatch = sqrt(info.mismatch)
    =#
    return
end