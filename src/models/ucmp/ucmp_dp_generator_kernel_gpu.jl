# for each generator (row), param will store the following:
# v_linear_cost[1:T], w_linear_cost[1:T], y_linear_cost[1:T]
# The linear cost captures quadratic penalty terms too (linearized for 0,1)

@inline function dp_generator_kernel(
    ngen::Int, T::Int,
    v0s::CuDeviceArray{Int64,1}, Tus::CuDeviceArray{Int64,1}, Tds::CuDeviceArray{Int64,1},
    Hus::CuDeviceArray{Int64,1}, Hds::CuDeviceArray{Int64,1},
    con::CuDeviceArray{Float64,1}, coff::CuDeviceArray{Float64,1},
    uc_u::CuDeviceArray{Float64,2}, uc_v::CuDeviceArray{Float64,2}, uc_z::CuDeviceArray{Float64,2},
    uc_l::CuDeviceArray{Float64,2}, uc_rho::CuDeviceArray{Float64,2},
    param::CuDeviceArray{Float64,2}
)
    I = blockIdx().x
    tx = threadIdx().x

    @inbounds begin
        if I <= ngen
            if tx == 1
                # const_cost = 0 # const_cost not needed for just finding the optimal UC scheduling
                lin_v_start, lin_w_start, lin_y_start = 0, T, 2*T
                for t in 1:T
                    param[I,lin_v_start+t] += uc_rho[I,3*t-2]/2 - uc_rho[I,3*t-2]*(uc_u[I,3*t-2]+uc_z[I,3*t-2]) - uc_l[I,3*t-2]
                    param[I,lin_w_start+t] += uc_rho[I,3*t-1]/2 - uc_rho[I,3*t-1]*(uc_u[I,3*t-1]+uc_z[I,3*t-1]) - uc_l[I,3*t-1]
                    param[I,lin_y_start+t] += uc_rho[I,3*t]/2 - uc_rho[I,3*t]*(uc_u[I,3*t]+uc_z[I,3*t]) - uc_l[I,3*t]
                end

                v0 = v0s[I]
                Tu = Tus[I]
                Td = Tds[I]
                Hu = Hus[I]
                Hd = Hds[I]

                # DP algorithm implementation
                sols = CuDynamicSharedArray(Int, 2*T)
                costs = CuDynamicSharedArray(Float64, 2*T, 2*T*sizeof(Int))
                for t in 1:2*T
                    sols[t] = 0
                    costs[t] = 0.0
                end

                for t in T:-1:1
                    if v0 == 1 && t <= Hu
                        if T > Hu
                            costs[v0+1] = costs[2*Hu+v0+1]
                            sols[v0+1] = sols[2*Hu+v0+1]
                        end

                        Hu = min(Hu, T)
                        for tt in 1:Hu
                            costs[v0+1] += param[I,lin_v_start+tt]
                            sols[v0+1] += v0 * 2^(T-tt)
                        end
                        break
                    elseif v0 == 0 && t <= Hd
                        if T > Hd
                            costs[v0+1] = costs[2*Hd+v0+1]
                            sols[v0+1] = sols[2*Hd+v0+1]
                        end
                        Hd = min(Hd, T)
                        for tt in 1:Hd
                            sols[v0+1] += v0 * 2^(T-tt)
                        end
                        break
                    end
                    for s in 0:1
                        Tforward = s == 1 ? Td : Tu
                        keep_cost = param[I,lin_v_start+t] * s + (t == T ? 0 : costs[2*t+s+1])
                        switch_lin_cost = 0
                        if s == 0
                            for tt in t:min(T, t+Tforward-1)
                                switch_lin_cost += param[I,lin_v_start+tt]
                            end
                        end
                        switch_cost = switch_lin_cost + (t+Tforward > T ? 0 : costs[2*(t+Tforward-1)+1-s+1]) + (s == 1 ? coff[I] + param[I,lin_y_start+t] : con[I] + param[I,lin_w_start+t])

                        if keep_cost < switch_cost
                            costs[2*(t-1)+s+1] = keep_cost
                            sols[2*(t-1)+s+1] = s * 2^(T-t)
                            if t < T
                                sols[2*(t-1)+s+1] += sols[2*t+s+1]
                            end
                        else
                            costs[2*(t-1)+s+1] = switch_cost
                            next_t = min(t+Tforward-1,T)
                            for tt in t:next_t
                                sols[2*(t-1)+s+1] += (1-s) * 2^(T-tt)
                            end
                            if next_t < T
                                sols[2*(t-1)+s+1] += sols[2*next_t+1-s+1]
                            end
                        end
                    end
                end


                # Decode the sols back to binary and update solution
                sol = sols[v0+1]
                for t in 1:T
                    uc_v[I,3*t-2] = (sol >> (T-t)) % 2
                end
                uc_v[I,2] = Int(uc_v[I,1] > v0)
                uc_v[I,3] = Int(uc_v[I,1] < v0)
                for t in 2:T
                    uc_v[I,3*t-1] = Int(uc_v[I,3*t-2] > uc_v[I,3*t-5])
                    uc_v[I,3*t] = Int(uc_v[I,3*t-2] < uc_v[I,3*t-5])
                end
            end
            CUDA.sync_threads()
        end
    end
    return
end