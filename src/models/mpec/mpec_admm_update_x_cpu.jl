function acopf_admm_update_x_gen(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ComplementarityModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    gen_sol::EmptyGeneratorSolution{Float64,Array{Float64,1}}
)
    grid, sol, info = mod.grid, mod.solution, mod.info

    baseMVA = grid.baseMVA
    ngen = grid.ngen
    gen_start = mod.gen_start
    pgmin, pgmax, qgmin, qgmax= mod.pgmin_curr, mod.pgmax_curr, grid.qgmin, grid.qgmax
    vgmin, vgmax, vm_setpoint = grid.vgmin, grid.vgmax, grid.vm_setpoint
    alpha, pg_setpoint = grid.alpha, grid.pg_setpoint
    c2, c1 = grid.c2, grid.c1

    u, v, z, l, rho = sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho

    # Voltage stability control via reactive power
    time_gen = @timed begin

        # Real power generations
        for I=1:ngen
            pg_idx = gen_start + 2*(I-1)
            u[pg_idx] = max(pgmin[I],
                            min(pgmax[I],
                                (-(c1[I]*baseMVA + l[pg_idx] + rho[pg_idx]*(-v[pg_idx] + z[pg_idx]))) / (2*c2[I]*(baseMVA^2) + rho[pg_idx])))

        end

        # Reactive power generation
        # There are three cases:
        #   i) qmin <= q <= qmax and v = vsp
        #  ii) q = qmin and v >= vsp
        # iii) q = qmax and v <= vsp

        qg_val = zeros(3)
        vg_val = zeros(3)
        obj_val = zeros(3)

        for I=1:ngen
            qg_idx = gen_start + 2*(I-1) + 1
            vg_idx = gen_start + 2*ngen + (I-1)
            qg_val[1] = max(qgmin[I],
                            min(qgmax[I],
                                (-(l[qg_idx] + rho[qg_idx]*(-v[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
            vg_val[1] = vm_setpoint[I]^2

            qg_val[2] = qgmin[I]
            vg_val[2] = max(max(vgmin[I]^2, vm_setpoint[I]^2),
                            (min(vgmax[I]^2,
                                 (-(l[vg_idx] + rho[vg_idx]*(-v[vg_idx] + z[vg_idx]))) / rho[vg_idx])))

            qg_val[3] = qgmax[I]
            vg_val[3] = max(vgmin[I]^2,
                            min(min(vgmax[I]^2, vm_setpoint[I]^2),
                                (-(l[vg_idx] + rho[vg_idx]*(-v[vg_idx] + z[vg_idx]))) / rho[vg_idx]))
            for i=1:3
                obj_val[i] = l[qg_idx]*(qg_val[i]-v[qg_idx]+z[qg_idx]) + (rho[qg_idx]/2)*(qg_val[i]-v[qg_idx]+z[qg_idx])^2 +
                             l[vg_idx]*(vg_val[i]-v[vg_idx]+z[vg_idx]) + (rho[vg_idx]/2)*(vg_val[i]-v[vg_idx]+z[vg_idx])^2
            end

            _, i = findmin(obj_val)
            u[qg_idx] = qg_val[i]
            u[vg_idx] = vg_val[i]
        end
    end

    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time

    # Frequency control via real power
    time_gen = @timed begin
        # Real power generation
        # There are three cases:
        #   i) pmin <= p <= pmax and p = psp + alpha*freq
        #  ii) p = pmin and p >= psp + alpha*freq
        # iii) p = pmax and p <= psp + alpha*freq

        pg_val = zeros(3)
        fg_val = zeros(3)
        obj_val = zeros(3)

        for I=1:ngen
            pg_idx = gen_start + 2*(I-1)
            fg_idx = gen_start + 3*ngen + (I-1)
            a = 2*c2[I]*(baseMVA*alpha[I])^2 + rho[pg_idx]*alpha[I]^2 + rho[fg_idx]
            b = 2*c2[I]*pg_setpoint[I]*(baseMVA)^2*alpha[I] + c1[I]*baseMVA*alpha[I] +
                l[pg_idx]*alpha[I] + rho[pg_idx]*(pg_setpoint[I]-v[pg_idx]+z[pg_idx])*alpha[I] +
                l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx])
            fg_val[1] = max((pgmax[I] - pg_setpoint[I])/alpha[I],
                            min((pgmin[I] - pg_setpoint[I])/alpha[I], -b/a))
            pg_val[1] = pg_setpoint[I] + alpha[I]*fg_val[1]

            pg_val[2] = pgmin[I]
            fg_val[2] = max((pgmin[I] - pg_setpoint[I])/alpha[I],
                            -(l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx]))/rho[fg_idx])

            pg_val[3] = pgmax[I]
            fg_val[3] = min((pgmax[I] - pg_setpoint[I])/alpha[I],
                             -(l[fg_idx] + rho[fg_idx]*(-v[fg_idx] + z[fg_idx]))/rho[fg_idx])

            for i=1:3
                obj_val[i] = c2[I]*(pg_val[i]*baseMVA)^2 + c1[I]*(pg_val[i]*baseMVA) +
                             l[pg_idx]*(pg_val[i]-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(pg_val[i]-v[pg_idx]+z[pg_idx])^2 +
                             l[fg_idx]*(fg_val[i]-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(fg_val[i]-v[fg_idx]+z[fg_idx])^2
            end

            _, i = findmin(obj_val)
            u[pg_idx] = pg_val[i]
            u[fg_idx] = fg_val[i]
        end
        # Reactive power generation
#=
        for I=1:ngen
            qg_idx = gen_start + 2*(I-1) + 1
            u[qg_idx] = max(qgmin[I],
                            min(qgmax[I],
                                (-(l[qg_idx] + rho[qg_idx]*(-v[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
        end
=#
    end

    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time

    # Storage model
    time_gen = @timed begin
        ps_val = zeros(2)
        obj_val = zeros(2)

        for I=1:grid.nstorage
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

            lb = max(grid.chg_min[I], (grid.energy_min[I]-grid.energy_setpoint[I])/grid.eta_chg[I])
            ub = min(grid.chg_max[I], (grid.energy_max[I]-grid.energy_setpoint[I])/grid.eta_chg[I])
            ps_val[1] = max(lb,
                            min(ub,
                                (-(l[sg_idx] + rho[sg_idx]*(-v[sg_idx] + z[sg_idx]))) / rho[sg_idx]))
            obj_val[1] = l[sg_idx]*(ps_val[1]-v[sg_idx]+z[sg_idx]) + (0.5*rho[sg_idx])*(ps_val[1]-v[sg_idx]+z[sg_idx])^2

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

            lb = max(grid.chg_min[I], (grid.energy_max[I]-grid.energy_setpoint[I])/(-grid.eta_dis[I]))
            ub = min(grid.chg_max[I], (grid.energy_min[I]-grid.energy_setpoint[I])/(-grid.eta_dis[I]))
            ps_val[2] = max(lb,
                            min(ub,
                                (l[sg_idx] + rho[sg_idx]*(-v[sg_idx] + z[sg_idx])) / rho[sg_idx]))
            obj_val[2] = l[sg_idx]*(-ps_val[2]-v[sg_idx]+z[sg_idx]) + (0.5*rho[sg_idx])*(-ps_val[2]-v[sg_idx]+z[sg_idx])^2

            _, i = findmin(obj_val)
            if i == 1
                u[sg_idx] = ps_val[i]
            else
                u[sg_idx] = -ps_val[i]
            end
#=
            mod_ipopt = JuMP.Model(Ipopt.Optimizer)
            @variable(mod_ipopt, grid.chg_min[I] <= CHG <= grid.chg_max[I], start=(grid.chg_min[I]+grid.chg_max[I])/2)
            @variable(mod_ipopt, grid.chg_min[I] <= DIS <= grid.chg_max[I], start=(grid.chg_min[I]+grid.chg_max[I])/2)
            @variable(mod_ipopt, PS, start=0.0)
            @variable(mod_ipopt, grid.energy_min[I] <= ENERGY <= grid.energy_max[I], start=grid.energy_setpoint[I])

            @constraint(mod_ipopt, pg_def,
                        PS == CHG - DIS
            )
            @constraint(mod_ipopt, energy_def,
                        ENERGY - grid.energy_setpoint[I] == (grid.eta_chg[I]*CHG - grid.eta_dis[I]*DIS)
            )
            @NLobjective(mod_ipopt, Min,
                         l[sg_idx]*(PS-v[sg_idx]+z[sg_idx]) + (0.5*rho[sg_idx])*(PS-v[sg_idx]+z[sg_idx])^2
            )
            JuMP.optimize!(mod_ipopt)
            if abs(u[sg_idx] - value(PS)) > 1e-6
                @printf("[%d] u = %.6e  PS = %.6e\n", sg_idx, u[sg_idx], value(PS))
            end
=#
        end
    end

    info.user.time_generators += time_gen.time
    info.time_x_update += time_gen.time

#=
    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        qg_idx = gen_start + 2*(g-1) + 1
        fg_idx = gen_start + 3*ngen + (g-1)

        mod_ipopt = JuMP.Model(Ipopt.Optimizer)
        @variable(mod_ipopt, pgmin[g] <= PG <= pgmax[g], start=(pgmin[g]+pgmax[g])/2)
        @variable(mod_ipopt, qgmin[g] <= QG <= qgmax[g], start=(qgmin[g]+qgmax[g])/2)
        @variable(mod_ipopt, FG, start=0.0)
        @variable(mod_ipopt, FG_W >= 0, start=0.0)
        @variable(mod_ipopt, FG_V >= 0, start=0.0)
        @constraint(mod_ipopt, Pg_setpoint,
            PG - (pg_setpoint[g] + alpha[g]*FG) == FG_W - FG_V
        )
        @NLconstraint(mod_ipopt, Fg_complementarity_w,
            (PG - pgmin[g])*(FG_W) <= 0.0
        )
        @NLconstraint(mod_ipopt, Fg_complementarity_v,
            (pgmax[g] - PG)*(FG_V) <= 0.0
        )
        @NLobjective(mod_ipopt, Min,
            c2[g]*(baseMVA*PG)^2 + c1[g]*(baseMVA*PG)
            + l[pg_idx]*(PG - v[pg_idx] + z[pg_idx])
            + (rho[pg_idx]/2)*(PG - v[pg_idx] + z[pg_idx])^2
#            + l[qg_idx]*(QG - v[qg_idx] + z[qg_idx])
#            + (rho[qg_idx]/2)*(QG - v[qg_idx] + z[qg_idx])^2
            + l[fg_idx]*(FG - v[fg_idx] + z[fg_idx])
            + (rho[fg_idx]/2)*(FG - v[fg_idx] + z[fg_idx])^2
        )
        JuMP.optimize!(mod_ipopt)

        obj_exa = c2[g]*(u[pg_idx]*baseMVA)^2 + c1[g]*(u[pg_idx]*baseMVA) +
                  l[pg_idx]*(u[pg_idx]-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(u[pg_idx]-v[pg_idx]+z[pg_idx])^2 +
                  l[fg_idx]*(u[fg_idx]-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(u[fg_idx]-v[fg_idx]+z[fg_idx])^2
        obj_ipopt = c2[g]*(JuMP.value(PG)*baseMVA)^2 + c1[g]*(JuMP.value(PG)*baseMVA) +
                    l[pg_idx]*(JuMP.value(PG)-v[pg_idx]+z[pg_idx]) + (rho[pg_idx]/2)*(JuMP.value(PG)-v[pg_idx]+z[pg_idx])^2 +
                    l[fg_idx]*(JuMP.value(FG)-v[fg_idx]+z[fg_idx]) + (rho[fg_idx]/2)*(JuMP.value(FG)-v[fg_idx]+z[fg_idx])^2

        @printf("[%3d]  PG(ExaTron) = % 12.6e  PG(Ipopt) = % 12.6e  diff = % 12.6e\n",
                g, u[pg_idx], JuMP.value(PG), abs(u[pg_idx] - JuMP.value(PG)))
        @printf("[%3d]  FG(ExaTron) = % 12.6e  FG(Ipopt) = % 12.6e  diff = % 12.6e\n",
                g, u[fg_idx], JuMP.value(FG), abs(u[fg_idx] - JuMP.value(FG)))

        if abs(u[pg_idx] - JuMP.value(PG)) > 1e-8
            @printf("[%3d]  PG(ExaTron) = % 12.6e  PG(Ipopt) = % 12.6e  diff = % 12.6e\n",
                    g, u[pg_idx], JuMP.value(PG), abs(u[pg_idx] - JuMP.value(PG)))
            @printf("       PG(ExaTron) == PG_setpoint + alpha * freq? % 12.6e == % 12.6e\n",
                    u[pg_idx], pg_setpoint[g]+alpha[g]*u[fg_idx])
            @printf("       PG(Ipopt)   == PG_setpoint + alpha * freq? % 12.6e == % 12.6e\n",
                    JuMP.value(PG), pg_setpoint[g]+alpha[g]*JuMP.value(FG))
        end
        if abs(obj_exa - obj_ipopt) > 1e-8
            @printf("[%3d] OBJ(ExaTron) = % 12.6e OBJ(Ipopt) = % 12.6e  diff = % 12.6e\n",
                    g, obj_exa, obj_ipopt, abs(obj_exa - obj_ipopt))

        end
        if abs(u[fg_idx] - JuMP.value(FG)) > 1e-8
            @printf("[%3d]  FG(ExaTron) = % 12.6e  FG(Ipopt) = % 12.6e  diff = % 12.6e\n",
                g, u[fg_idx], JuMP.value(FG), abs(u[fg_idx] - JuMP.value(FG)))
        end
    end
=#
#=
    for g=1:ngen
        pg_idx = gen_start + 2*(g-1)
        qg_idx = gen_start + 2*(g-1) + 1
        vg_idx = gen_start + 2*ngen + (g-1)

        mod_ipopt = JuMP.Model(Ipopt.Optimizer)
        @variable(mod_ipopt, pgmin[g] <= Pg <= pgmax[g], start=(pgmin[g]+pgmax[g])/2)
        @variable(mod_ipopt, qgmin[g] <= Qg <= qgmax[g], start=(qgmin[g]+qgmax[g])/2)
        @variable(mod_ipopt, vgmin[g]^2 <= Vg <= vgmax[g]^2, start=((vgmin[g]+vgmax[g])/2)^2)
        @variable(mod_ipopt, Vg_w >= 0, start=0.0)
        @variable(mod_ipopt, Vg_v >= 0, start=0.0)
        @constraint(mod_ipopt, Vg_setpoint,
            Vg - vm_setpoint[g]^2 == Vg_w - Vg_v
        )
        @NLconstraint(mod_ipopt, Vg_complementarity_w,
            (Qg - qgmin[g])*(Vg_w) <= 0.0
        )
        @NLconstraint(mod_ipopt, Vg_complementarity_v,
            (qgmax[g] - Qg)*(Vg_v) <= 0.0
        )
        @NLobjective(mod_ipopt, Min,
#            c2[g]*(baseMVA*Pg)^2 + c1[g]*(baseMVA*Pg)
#            + l[pg_idx]*(Pg - v[pg_idx] + z[pg_idx])
#            + (rho[pg_idx]/2)*(Pg - v[pg_idx] + z[pg_idx])^2
            + l[qg_idx]*(Qg - v[qg_idx] + z[qg_idx])
            + (rho[qg_idx]/2)*(Qg - v[qg_idx] + z[qg_idx])^2
            + l[vg_idx]*(Vg - v[vg_idx] + z[vg_idx])
            + (rho[vg_idx]/2)*(Vg - v[vg_idx] + z[vg_idx])^2
        )
        JuMP.optimize!(mod_ipopt)
#=
        if abs(u[pg_idx] - JuMP.value(Pg)) > 1e-8
            @printf("[%3d] Pg(ExaTron) = % 12.6e  Pg(Ipopt) = % 12.6e  diff = % 12.6e\n",
                g, u[pg_idx], JuMP.value(Pg), abs(u[pg_idx] - JuMP.value(Pg)))
        end
=#
        if abs(u[qg_idx] - JuMP.value(Qg)) > 1e-6 || abs(u[vg_idx] - JuMP.value(Vg)) > 1e-6
            obj_exa = l[qg_idx]*(u[qg_idx]-v[qg_idx]+z[qg_idx]) + (rho[qg_idx]/2)*(u[qg_idx]-v[qg_idx]+z[qg_idx])^2 +
                      l[vg_idx]*(u[vg_idx]-v[vg_idx]+z[vg_idx]) + (rho[vg_idx]/2)*(u[vg_idx]-v[vg_idx]+z[vg_idx])^2
            obj_ipopt = l[qg_idx]*(JuMP.value(Qg) - v[qg_idx] + z[qg_idx]) +
                        (rho[qg_idx]/2)*(JuMP.value(Qg) - v[qg_idx] + z[qg_idx])^2 +
                        l[vg_idx]*(JuMP.value(Vg) - v[vg_idx] + z[vg_idx]) +
                        (rho[vg_idx]/2)*(JuMP.value(Vg) - v[vg_idx] + z[vg_idx])^2

            @printf("[%3d] [% 12.6e,% 12.6e] Qg(ExaTron)  = % 12.6e  Qg(Ipopt)  = % 12.6e  diff = % 12.6e\n",
                g, qgmin[g], qgmax[g], u[qg_idx], JuMP.value(Qg), abs(u[qg_idx] - JuMP.value(Qg)))
            @printf("      [% 12.6e,% 12.6e] Vg(ExaTron)  = % 12.6e  Vg(Ipopt)  = % 12.6e  diff = % 12.6e\n",
                vgmin[g]^2, vgmax[g]^2, u[vg_idx], JuMP.value(Vg), abs(u[vg_idx] - JuMP.value(Vg)))
            @printf("     Obj(ExaTron)  = % 12.6e  Obj(Ipopt) = % 12.6e  diff = % 12.6e\n",
                obj_exa, obj_ipopt, abs(obj_exa - obj_ipopt))
            @printf("%s\n", objective_function_string(IJuliaMode, mod_ipopt))
        end

        #=
        obj = l[qg_idx]*(JuMP.value(Qg) - v[qg_idx] + z[qg_idx]) +
              (rho[qg_idx]/2)*(JuMP.value(Qg) - v[qg_idx] + z[qg_idx])^2 +
              l[vg_idx]*(JuMP.value(Vg) - v[vg_idx] + z[vg_idx]) +
              (rho[vg_idx]/2)*(JuMP.value(Vg) - v[vg_idx] + z[vg_idx])^2
        @printf("[%3d] Obj = % 12.6e  Pg = % 12.6e  Qg = % 12.6e [% 12.6e,% 12.6e]  Vg = %12.6e  %s\n",
            g, obj,
            JuMP.value(Pg),
            JuMP.value(Qg), qgmin[g], qgmax[g],
            JuMP.value(Vg),
            termination_status(mod_ipopt)
        )
        =#
    end
=#
    return
end

function acopf_admm_update_x_line(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ComplementarityModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    par, grid, sol, info = env.params, mod.grid, mod.solution, mod.info

    if env.use_linelimit
        time_br = @timed auglag_it, tron_it = auglag_linelimit_two_level_alternative(mod.n, grid.nline, mod.line_start,
                                                info.inner, par.max_auglag, par.mu_max, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, grid.YffR, grid.YffI, grid.YftR, grid.YftI,
                                                grid.YttR, grid.YttI, grid.YtfR, grid.YtfI,
                                                grid.FrVmBound, grid.ToVmBound, grid.FrVaBound, grid.ToVaBound)
    else
        time_br = @timed auglag_it, tron_it = polar_kernel_two_level_alternative(mod.n, grid.nline, mod.line_start, par.scale,
                                                sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                par.shift_lines, mod.membuf, grid.YffR, grid.YffI, grid.YftR, grid.YftI,
                                                grid.YttR, grid.YttI, grid.YtfR, grid.YtfI, grid.FrVmBound, grid.ToVmBound)
    end

    info.user.time_branches += time_br.time
    info.time_x_update += time_br.time
    return
end

function acopf_admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ComplementarityModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
    acopf_admm_update_x_gen(env, mod, mod.gen_solution)
    acopf_admm_update_x_line(env, mod)
end