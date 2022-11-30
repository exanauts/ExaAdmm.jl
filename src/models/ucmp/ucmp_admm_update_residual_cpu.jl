"""
Compute and update the primal and dual residuals at current iteration.

### Arguments

- `env::AdmmEnv` -- struct that defines the environment of ADMM
- `mod::UCMPModel` -- struct that defines model

### Notes

The primal and dual residuals are stored in `mod.solution.rp` and `mod.solution.rd`, respectively.
"""
function admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    info = mod.info

    info.primres = 0.0
    info.dualres = 0.0
    info.norm_z_curr = 0.0
    info.mismatch = 0.0

    admm_update_residual(env, mod.mpmodel, device)

    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution

    # update residuals from uc solution, these residuals will be added to each model at time t
    uc_sol.rp .= uc_sol.u_curr .- uc_sol.v_curr .+ uc_sol.z_curr
    uc_sol.rd .= uc_sol.z_curr .- uc_sol.z_prev
    uc_sol.Ax_plus_By .= uc_sol.rp .- uc_sol.z_curr
    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        model.info.primres = sqrt(model.info.primres^2 + norm(uc_sol.rp[:, 3*i-2:3*i])^2)
        model.info.dualres = sqrt(model.info.dualres^2 + norm(uc_sol.rd[:, 3*i-2:3*i])^2)
        model.info.norm_z_curr = sqrt(model.info.norm_z_curr^2 + norm(uc_sol.z_curr[:, 3*i-2:3*i])^2)
        model.info.mismatch = sqrt(model.info.mismatch^2 + norm(uc_sol.Ax_plus_By[:, 3*i-2:3*i])^2)
    end


    # update residuals from v ramp solution, these residuals will be added to each model at time t
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        model = mod.mpmodel.models[i]
        v_curr = @view uc_sol.v_curr[:, 3*i-2]
        vr_sol.rp .= vr_sol.u_curr .- v_curr .+ vr_sol.z_curr
        vr_sol.rd .= vr_sol.z_curr .- vr_sol.z_prev
        vr_sol.Ax_plus_By .= vr_sol.rp .- vr_sol.z_curr

        model.info.primres = sqrt(model.info.primres^2 + norm(vr_sol.rp)^2)
        model.info.dualres = sqrt(model.info.dualres^2 + norm(vr_sol.rd)^2)
        model.info.norm_z_curr = sqrt(model.info.norm_z_curr^2 + norm(vr_sol.z_curr)^2)
        model.info.mismatch = sqrt(model.info.mismatch^2 + norm(vr_sol.Ax_plus_By)^2)
    end

    for i=1:mod.mpmodel.len_horizon
        info.primres = max(info.primres, mod.mpmodel.models[i].info.primres)
        info.dualres = max(info.dualres, mod.mpmodel.models[i].info.dualres)
        info.norm_z_curr = max(info.norm_z_curr, mod.mpmodel.models[i].info.norm_z_curr)
        info.mismatch = max(info.mismatch, mod.mpmodel.models[i].info.mismatch)
    end

    return
end
