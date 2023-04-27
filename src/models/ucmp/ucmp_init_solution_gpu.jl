# TODO: initialization for uc_sol.u_curr seems meaningless here... will skip for now
function uc_init_solution_kernel(i::Int, ngen::Int, gen_start::Int,
    pgmax::CuDeviceArray{Float64,1}, pgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1}, qgmin::CuDeviceArray{Float64,1},
    R::CuDeviceArray{Float64,1},
    u::CuDeviceArray{Float64,1}, v_prev::CuDeviceArray{Float64,1}, uc_v::CuDeviceArray{Float64,2},
    r_u::CuDeviceArray{Float64,1}, vr_u::CuDeviceArray{Float64,1}, 
    r_s::CuDeviceArray{Float64,1}, uc_u::CuDeviceArray{Float64,2}, uc_s::CuDeviceArray{Float64,2}
)
    g = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if g <= ngen
        Ri = R[g] # Assume RU = RD = Ri
        Si = Ri # Assume SU = SD = Si. TODO: account for startup/shutdown ramping later. For now assume Si = Ri.
        p_curr = u[gen_start+2*(g-1)]
        q_curr = u[gen_start+2*(g-1)+1]
        v, w, y = uc_v[g, 3*i-2], uc_v[g, 3*i-1], uc_v[g, 3*i]
        if i > 1
            p_prev = v_prev[gen_start+2*(g-1)]
            vhat = uc_v[g, 3*(i-1)-2]
            r_u[g] = v_prev[gen_start+2*(g-1)]
            vr_u[g] = vhat
            r_s[2*g-1] = p_curr - p_prev - Ri*vhat - Si*w
            r_s[2*g] = -(p_curr - p_prev + Ri*v + Si*y)    
        end
        Pmax, Pmin, Qmax, Qmin = pgmax[g], pgmin[g], qgmax[g], qgmin[g]
        uc_u[g, 3*i-2] = v
        uc_u[g, 3*i-1] = w
        uc_u[g, 3*i] = y
        # s_curr: [s^{p,UB}, s^{p,LB}, s^{q,UB}, s^{q,LB}]
        uc_s[g, 4*i-3] = p_curr - Pmax*v 
        uc_s[g, 4*i-2] = -(p_curr - Pmin*v)
        uc_s[g, 4*i-1] = q_curr - Qmax*v
        uc_s[g, 4*i] = -(q_curr - Qmin*v)
    end
    return
end


function uc_init_mpmodel_solution!(
    mod::ModelMpacopf{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol::Vector{SolutionRamping{Float64,CuArray{Float64,1}}},
    vr_sol::Vector{SolutionRamping{Float64,CuArray{Float64,1}}},
    uc_sol::SolutionUC{Float64,CuArray{Float64,2}},
    rho_pq::Float64, rho_va::Float64
)
    for i=1:mod.len_horizon
        init_solution!(mod.models[i], mod.models[i].solution, rho_pq, rho_va)
        fill!(sol[i], 0.0)
        sol[i].rho .= rho_pq
        griddata = mod.models[1].grid_data

        @inbounds begin
            CUDA.@timed @cuda threads=64 blocks=(div(griddata.ngen-1,64)+1) uc_init_solution_kernel(
                i, griddata.ngen, mod.models[i].gen_start, 
                griddata.pgmax, griddata.pgmin, griddata.qgmax, griddata.qgmin, 
                griddata.ramp_rate, mod.models[i].solution.u_curr, mod.models[max(i-1, 1)].solution.v_curr, uc_sol.v_curr, sol[i].u_curr,
                vr_sol[i].u_curr, sol[i].s_curr, uc_sol.u_curr, uc_sol.s_curr
            )
        end
    end

end


function init_solution!(
    mod::UCMPModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    sol::SolutionUC{Float64,CuArray{Float64,2}},
    rho_pq::Float64, rho_va::Float64, rho_uc=40000.0::Float64
)
    vr_solution = mod.vr_solution
    for i=2:mod.mpmodel.len_horizon
        vr_solution[i].rho .= rho_uc
    end
    uc_init_mpmodel_solution!(mod.mpmodel, mod.mpmodel.solution, vr_solution, sol, rho_pq, rho_va)
    sol.rho .= rho_uc
end