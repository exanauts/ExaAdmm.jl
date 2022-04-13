function mpec_storage_model_kernel(
    nsto::Int, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    chg_min::CuDeviceArray{Float64,1}, chg_max::CuDeviceArray{Float64,1},
    energy_min::CuDeviceArray{Float64,1}, energy_max::CuDeviceArray{Float64,1},
    energy_setpoint::CuDeviceArray{Float64,1},
    eta_chg::CuDeviceArray{Float64,1}, eta_dis::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= nsto
        sg_idx = gen_start + 4*ngen + (I-1)

        # e_s - e^setpoint_s = (delta t)*(eta_chg*p^chg_s - eta_dis*p^dis_s)
        # We assume delta t = 1.

        # Case 1: discharge = 0
        # We charge the storage system in this case.
        # e^l_s <= e_s = eta_chg*p^chg_s + e^setpoint_s <= e^u_s
        # p^l_s <= p^chg_s <= p^u_s
        # ==>
        #  lb = max(p^l_s, (e^l_s - e^setpoint_s)/eta_chg)
        #  ub = min(p^u_s, (e^u_s - e^setpoint_s)/eta_chg)
        #
        # We minimize over p^chg_s under [lb,ub] with an objective function:
        #  l_s*(p^chg_s - v_s + z_s) + (0.5*rho_s)*(p^chg_s - v_s + z_s)^2

        lb = max(chg_min[I], (energy_min[I]-energy_setpoint[I])/eta_chg[I])
        ub = min(chg_max[I], (energy_max[I]-energy_setpoint[I])/eta_chg[I])
        ps_val1 = max(lb,
                      min(ub,
                          (-(l[sg_idx] + rho[sg_idx]*(-v[sg_idx] + z[sg_idx]))) / rho[sg_idx]))
        obj_val1 = l[sg_idx]*(ps_val1-v[sg_idx]+z[sg_idx]) + (0.5*rho[sg_idx])*(ps_val1-v[sg_idx]+z[sg_idx])^2

        # Case 2: charge = 0
        # We discharge the storage system in this case.
        # e^l_s <= e_s = -eta_dis*p^dis_s + e^setpoint_s <= e^u_s
        # p^l_s <= p^dis_s <= p^u_s
        # ==>
        #  lb = max(p^l_s, (e^u_s - e^setpoint_s)/(-eta_dis))
        #  ub = min(p^u_s, (e^l_s - e^setpoint_s)/(-eta_dis))
        #
        # We minimize over p^dis_s under [lb,ub] with an objective function:
        #  l_s*(-p^dis_s - v_s + z_s) + (0.5*rho_s)*(-p^dis_s - v_s + z_s)^2

        lb = max(chg_min[I], (energy_max[I]-energy_setpoint[I])/(-eta_dis[I]))
        ub = min(chg_max[I], (energy_min[I]-energy_setpoint[I])/(-eta_dis[I]))
        ps_val2 = max(lb,
                      min(ub,
                         (l[sg_idx] + rho[sg_idx]*(-v[sg_idx] + z[sg_idx])) / rho[sg_idx]))
        obj_val2 = l[sg_idx]*(-ps_val2-v[sg_idx]+z[sg_idx]) + (0.5*rho[sg_idx])*(-ps_val2-v[sg_idx]+z[sg_idx])^2

        if obj_val1 <= obj_val2
            u[sg_idx] = ps_val1
        else
            u[sg_idx] = -ps_val2
        end
    end
    return
end

function mpec_frequency_control_kernel(
    baseMVA::Float64, ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    pg_setpoint::CuDeviceArray{Float64,1}, alpha::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= ngen
        # Primary frequency control.
        pg_idx = gen_start + 2*(I-1)
        fg_idx = gen_start + 3*ngen + (I-1)
        a = 2*c2[I]*(baseMVA*alpha[I])^2 + rho[pg_idx]*alpha[I]^2 + rho[fg_idx]
        b = 2*c2[I]*pg_setpoint[I]*(baseMVA)^2*alpha[I] + c1[I]*baseMVA*alpha[I] +
            l[pg_idx]*alpha[I] + rho[pg_idx]*(pg_setpoint[I]-v[pg_idx]+z[pg_idx])*alpha[I] +
            l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx])
        fg_val1 = max((pgmax[I] - pg_setpoint[I])/alpha[I],
                      min((pgmin[I] - pg_setpoint[I])/alpha[I], -b/a))
        pg_val1 = pg_setpoint[I] + alpha[I]*fg_val1
        obj_val1 = c2[I]*(pg_val1*baseMVA)^2 + c1[I]*(pg_val1*baseMVA) +
                   l[pg_idx]*(pg_val1-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(pg_val1-v[pg_idx]+z[pg_idx])^2 +
                   l[fg_idx]*(fg_val1-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(fg_val1-v[fg_idx]+z[fg_idx])^2

        pg_val2 = pgmin[I]
        fg_val2 = max((pgmin[I] - pg_setpoint[I])/alpha[I],
                       -(l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx]))/rho[fg_idx])
        obj_val2 = c2[I]*(pg_val2*baseMVA)^2 + c1[I]*(pg_val2*baseMVA) +
                   l[pg_idx]*(pg_val2-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(pg_val2-v[pg_idx]+z[pg_idx])^2 +
                   l[fg_idx]*(fg_val2-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(fg_val2-v[fg_idx]+z[fg_idx])^2

        pg_val3 = pgmax[I]
        fg_val3 = min((pgmax[I] - pg_setpoint[I])/alpha[I],
                       -(l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx]))/rho[fg_idx])
        obj_val3 = c2[I]*(pg_val3*baseMVA)^2 + c1[I]*(pg_val3*baseMVA) +
                   l[pg_idx]*(pg_val3-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(pg_val3-v[pg_idx]+z[pg_idx])^2 +
                   l[fg_idx]*(fg_val3-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(fg_val3-v[fg_idx]+z[fg_idx])^2

        if obj_val1 <= obj_val2
            if obj_val1 <= obj_val3
                u[pg_idx] = pg_val1
                u[fg_idx] = fg_val1
            else
                u[pg_idx] = pg_val3
                u[fg_idx] = fg_val3
            end
        else
            if obj_val2 <= obj_val3
                u[pg_idx] = pg_val2
                u[fg_idx] = fg_val2
            else
                u[pg_idx] = pg_val3
                u[fg_idx] = fg_val3
            end
        end
    end
    return
end

function mpec_voltage_stability_kernel(
    ngen::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    vgmin::CuDeviceArray{Float64,1}, vgmax::CuDeviceArray{Float64,1}, vm_setpoint::CuDeviceArray{Float64,1},
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= ngen
        # Voltage stability control via reactive power.
        qg_idx = gen_start + 2*(I-1) + 1
        vg_idx = gen_start + 2*ngen + (I-1)

        qg_val1 = max(qgmin[I],
                      min(qgmax[I],
                          (-(l[qg_idx] + rho[qg_idx]*(-v[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
        vg_val1 = vm_setpoint[I]^2
        obj_val1 = l[qg_idx]*(qg_val1-v[qg_idx]+z[qg_idx]) + (rho[qg_idx]/2)*(qg_val1-v[qg_idx]+z[qg_idx])^2 +
                   l[vg_idx]*(vg_val1-v[vg_idx]+z[vg_idx]) + (rho[vg_idx]/2)*(vg_val1-v[vg_idx]+z[vg_idx])^2

        qg_val2 = qgmin[I]
        vg_val2 = max(max(vgmin[I]^2, vm_setpoint[I]^2),
                         (min(vgmax[I]^2,
                              (-(l[vg_idx] + rho[vg_idx]*(-v[vg_idx] + z[vg_idx]))) / rho[vg_idx])))
        obj_val2 = l[qg_idx]*(qg_val2-v[qg_idx]+z[qg_idx]) + (rho[qg_idx]/2)*(qg_val2-v[qg_idx]+z[qg_idx])^2 +
                   l[vg_idx]*(vg_val2-v[vg_idx]+z[vg_idx]) + (rho[vg_idx]/2)*(vg_val2-v[vg_idx]+z[vg_idx])^2

        qg_val3 = qgmax[I]
        vg_val3 = max(vgmin[I]^2,
                      min(min(vgmax[I]^2, vm_setpoint[I]^2),
                              (-(l[vg_idx] + rho[vg_idx]*(-v[vg_idx] + z[vg_idx]))) / rho[vg_idx]))
        obj_val3 = l[qg_idx]*(qg_val3-v[qg_idx]+z[qg_idx]) + (rho[qg_idx]/2)*(qg_val3-v[qg_idx]+z[qg_idx])^2 +
                   l[vg_idx]*(vg_val3-v[vg_idx]+z[vg_idx]) + (rho[vg_idx]/2)*(vg_val3-v[vg_idx]+z[vg_idx])^2

        if obj_val1 <= obj_val2
            if obj_val1 <= obj_val3
                u[qg_idx] = qg_val1
                u[vg_idx] = vg_val1
            else
                u[qg_idx] = qg_val3
                u[vg_idx] = vg_val3
            end
        else
            if obj_val2 <= obj_val3
                u[qg_idx] = qg_val2
                u[vg_idx] = vg_val2
            else
                u[qg_idx] = qg_val3
                u[vg_idx] = vg_val3
            end
        end
    end
    return
end

function acopf_admm_update_x_gen(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    gen_sol::EmptyGeneratorSolution{Float64,CuArray{Float64,1}}
)
    grid, sol, info = mod.grid, mod.solution, mod.info
    nblk_gen = div(grid.ngen-1, 32)+1

    time_gen = @timed begin
        @cuda threads=32 blocks=nblk_gen mpec_voltage_stability_kernel(
            grid.ngen, mod.gen_start, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            grid.qgmin, grid.qgmax, grid.vgmin, grid.vgmax, grid.vm_setpoint
        )
        @cuda threads=32 blocks=nblk_gen mpec_frequency_control_kernel(
            grid.baseMVA, grid.ngen, mod.gen_start, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            grid.pgmin, grid.pgmax, grid.pg_setpoint, grid.alpha, grid.c2, grid.c1
        )

        if grid.nstorage > 0
            nblk_sto = div(grid.nstorage-1, 32)+1
            @cuda threads=32 blocks=nblk_sto mpec_storage_model_kernel(
                grid.nstorage, grid.ngen, mod.gen_start,
                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                grid.chg_min, grid.chg_max, grid.energy_min, grid.energy_max, grid.energy_setpoint,
                grid.eta_chg, grid.eta_dis
            )
        end

        CUDA.synchronize()
    end

    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time
    return
end

function acopf_admm_update_x_line(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, grid, sol, info = env.params, mod.grid, mod.solution, mod.info
    shmem_size = env.params.shmem_size

    if env.use_linelimit
        time_br = CUDA.@timed @cuda threads=32 blocks=grid.nline shmem=shmem_size auglag_linelimit_two_level_alternative(
            mod.n, grid.nline, mod.line_start,
            info.inner, par.max_auglag, par.mu_max, par.scale,
            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            par.shift_lines, mod.membuf, grid.YffR, grid.YffI, grid.YftR, grid.YftI,
            grid.YttR, grid.YttI, grid.YtfR, grid.YtfI,
            grid.FrVmBound, grid.ToVmBound, grid.FrVaBound, grid.ToVaBound
        )
    else
        time_br = CUDA.@timed @cuda threads=32 blocks=grid.nline shmem=shmem_size polar_kernel_two_level_alternative(
            mod.n, grid.nline, mod.line_start, par.scale,
            sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
            par.shift_lines, mod.membuf, grid.YffR, grid.YffI, grid.YftR, grid.YftI,
            grid.YttR, grid.YttI, grid.YtfR, grid.YtfI, grid.FrVmBound, grid.ToVmBound
        )
    end

    info.time_x_update += time_br.time
    info.user.time_branches += time_br.time
    return
end

function acopf_admm_update_x(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ComplementarityModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    acopf_admm_update_x_gen(env, mod, mod.gen_solution)
    acopf_admm_update_x_line(env, mod)
end