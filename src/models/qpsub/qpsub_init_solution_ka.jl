"""
    init_solution()

- initialize sol.v_curr and sol.rho for all coupling
- initialize sqp_line, supY
"""

@kernel function init_generator_kernel_qpsub_ka(n::Int, gen_start::Int,
    pgmax, pgmin,
    qgmax, qgmin,
    v
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    g = J + (@groupsize()[1] * (I - 1))
    if g <= n
        v[gen_start + 2*(g-1)] = 0.5*(pgmin[g] + pgmax[g])
        v[gen_start + 2*(g-1)+1] = 0.5*(qgmin[g] + qgmax[g])
    end
end


@kernel function init_branch_bus_kernel_qpsub_ka(n::Int, line_start::Int, rho_va::Float64,
    YffR, YffI,
    YftR, YftI,
    YtfR, YtfI,
    YttR, YttI,
    us, ls, sqp_line,
    v, rho, supY, warm_start::Bool
)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    l = J + (@groupsize()[1] * (I - 1))
    if l <= n
        sqp_line[1,l] = (ls[l,1] + us[l,1])/2  # order |w_ijR  | w_ijI |  wi(ij) | wj(ji) |  thetai(ij) |  thetaj(ji)|
        sqp_line[2,l] = (ls[l,2] + us[l,2])/2
        sqp_line[3,l] = (ls[l,3] + us[l,3])/2
        sqp_line[4,l] = (ls[l,4] + us[l,4])/2
        sqp_line[5,l] = (ls[l,5] + us[l,5])/2
        sqp_line[6,l] = (ls[l,6] + us[l,6])/2

        pij_idx = line_start + 8*(l-1)

        supY[4*(l-1) + 1,3] = YftR[l]
        supY[4*(l-1) + 1,4] = YftI[l]
        supY[4*(l-1) + 1,5] = YffR[l]
        supY[4*(l-1) + 2,3] = -YftI[l]
        supY[4*(l-1) + 2,4] = YftR[l]
        supY[4*(l-1) + 2,5] = -YffI[l]
        supY[4*(l-1) + 3,3] = YtfR[l]
        supY[4*(l-1) + 3,4] = -YtfI[l]
        supY[4*(l-1) + 3,6] = YttR[l]
        supY[4*(l-1) + 4,3] = -YtfI[l]
        supY[4*(l-1) + 4,4] = -YtfR[l]
        supY[4*(l-1) + 4,6] = -YttI[l]


        if !warm_start
            v[pij_idx] = YftR[l]*sqp_line[1,l] + YftI[l]*sqp_line[2,l] + YffR[l]*sqp_line[3,l]  #CUBLAS.dot(4, supY[1,:],sqp_line[:,l])  #p_ij
            v[pij_idx+1] = -YftI[l]*sqp_line[1,l] + YftR[l]*sqp_line[2,l] - YffI[l]*sqp_line[3,l] #CUBLAS.dot(4, supY[2,:],sqp_line[:,l]) #q_ij
            v[pij_idx+2] = YtfR[l]*sqp_line[1,l] - YtfI[l]*sqp_line[2,l] + YttR[l]*sqp_line[4,l] #CUBLAS.dot(4, supY[3,:],sqp_line[:,l]) #p_ji
            v[pij_idx+3] = -YtfI[l]*sqp_line[1,l] - YtfR[l]*sqp_line[2,l] - YttI[l]*sqp_line[4,l] #CUBLAS.dot(4, supY[4,:],sqp_line[:,l]) #q_ji
            v[pij_idx+4] = sqp_line[3,l] #w_i
            v[pij_idx+5] = sqp_line[4,l] #w_j
            v[pij_idx+6] = sqp_line[5,l] #theta_i
            v[pij_idx+7] = sqp_line[6,l] #theta_j

            rho[pij_idx:pij_idx+7] .= rho_va
        end

    end
end

function init_solution!(
    model::ModelQpsub,
    sol::Solution,
    rho_pq::Float64, rho_va::Float64, warm_start::Bool,
    device
    )

    data = model.grid_data

    if !warm_start
        fill!(sol, 0.0)
        fill!(model.lambda, 0.0)

        #qpsub var
        sol.rho .= rho_pq


        nblk_gen = div(data.ngen-1,64)+1
        ev = init_generator_kernel_qpsub_ka(device, 64, 64*nblk_gen)(data.ngen, model.gen_start,
                        model.qpsub_pgmax, model.qpsub_pgmin, model.qpsub_qgmax, model.qpsub_qgmin, sol.v_curr)
    end

    nblk_line = div(data.nline-1,64)+1
    ev = init_branch_bus_kernel_qpsub_ka(device, 64, 64*nblk_line)(data.nline, model.line_start, rho_va,
                    data.YffR, data.YffI, data.YftR, data.YftI,
                    data.YtfR, data.YtfI, data.YttR, data.YttI, model.us, model.ls, model.sqp_line, sol.v_curr, sol.rho, model.supY, warm_start)
    KA.synchronize(device)

    return
end
