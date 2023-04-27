function admm_outer_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    admm_outer_prestep(env, mod.mpmodel, device)
    
    # v ramp consensus constraints.
    vr_sols = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        vr_sol = vr_sols[i]
        model.info.norm_z_prev = sqrt(model.info.norm_z_prev^2 + CUDA.norm(vr_sol.z_curr)^2)
    end

    # uc variable consensus constraints
    uc_sol = mod.uc_solution
    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        model.info.norm_z_prev = sqrt(model.info.norm_z_prev^2 + CUDA.norm(uc_sol.z_curr[:, 3*i-2:3*i])^2)
    end

    # update info for the UCMPModel level
    info = mod.info
    info.norm_z_prev = 0.0
    for i=1:mod.mpmodel.len_horizon
        model = mod.mpmodel.models[i]
        info.norm_z_prev = max(info.norm_z_prev, model.info.norm_z_prev)
    end

    return
end

"""
Implement any algorithmic steps required before each inner iteration.
"""
function admm_inner_prestep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    admm_inner_prestep(env, mod.mpmodel, device)
    uc_sol, vr_sols = mod.uc_solution, mod.vr_solution
    copyto!(uc_sol.z_prev, uc_sol.z_curr)
    for i=2:mod.mpmodel.len_horizon
        vr_sol = vr_sols[i]
        copyto!(vr_sol.z_prev, vr_sol.z_curr)
    end
    return
end

"""
Implement any steps required after the algorithm terminates.
"""
function calculate_on_off_objective(t::Int, ngen::Int, 
    x::CuDeviceArray{Float64, 1}, uc_sol::CuDeviceArray{Float64, 2}, on_cost::CuDeviceArray{Float64, 1}, off_cost::CuDeviceArray{Float64, 1}
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if tx <= ngen
        x[tx] += uc_sol[tx, 3*t-1] * on_cost[tx] + uc_sol[tx, 3*t] * off_cost[tx]
    end
    return
end

function admm_poststep(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    info = mod.info
    admm_poststep(env, mod.mpmodel, device) # NOTE: the calculation for info.user.err_ramp (ramp violation) in this step is wrong for UCMP, avoid using this
    info.objval = mod.mpmodel.info.objval
    ngen = mod.mpmodel.models[1].grid_data.ngen
    x = CUDA.fill(0.0, ngen)
    on_cost, off_cost = mod.uc_params.con, mod.uc_params.coff
    # add uc objective terms
    for i=1:mod.mpmodel.len_horizon
        CUDA.@sync @cuda threads=64 blocks=(div(ngen-1,64)+1) calculate_on_off_objective(i, ngen, x, mod.uc_solution.u_curr, on_cost, off_cost)
    end
    info.objval += sum(x)
    return
end