function eval_f_generator_kernel_cpu(
    I::Int, x::Array{Float64,1}, param::Array{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    f = 0.0

    @inbounds begin
        # x[1]      : p_{t,I}
        # x[2]      : phat_{t-1,I}
        # x[3]      : s_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{phat_{t-1,I}}
        # param[3,I]: rho_{p_{t,I}}
        # param[4,I]: rho_{phat_{t-1,I}}
        # param[5,I]: pbar_{t,I} - z_{p_{t,I}}
        # param[6,I]: pbar_{t-1,I} - z_{phat_{t-1,I}}
        # param[7,I]: mu
        # param[8,I]: xi

        f += c2*(x[1]*baseMVA)^2 + c1*(x[1]*baseMVA) + c0
        f += param[1,I]*(x[1] - param[5,I]) + (0.5*param[3,I])*(x[1] - param[5,I])^2
        f += param[2,I]*(x[2] - param[6,I]) + (0.5*param[4,I])*(x[2] - param[6,I])^2
        f += param[7,I]*(x[1] - x[2] - x[3]) + (0.5*param[8,I])*(x[1] - x[2] - x[3])^2
    end

    f *= scale
    return f
end

function eval_grad_f_generator_kernel_cpu(
    I::Int, x::Array{Float64,1}, g::Array{Float64,1}, param::Array{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        g[1] = 2*c2*(baseMVA)^2*x[1] + c1*baseMVA
        g[1] += param[1,I] + param[3,I]*(x[1] - param[5,I])
        g[1] += param[7,I] + param[8,I]*(x[1] - x[2] - x[3])
        g[1] *= scale
        g[2] = param[2,I] + param[4,I]*(x[2] - param[6,I])
        g[2] += -(param[7,I] + param[8,I]*(x[1] - x[2] - x[3]))
        g[2] *= scale
        g[3] = -(param[7,I] + param[8,I]*(x[1] - x[2] - x[3]))
        g[3] *= scale
    end
    return
end

function eval_h_generator_kernel_cpu(
    I::Int, x::Array{Float64,1}, mode::Symbol, scale::Float64,
    rows, cols, lambda::Array{Float64,1}, values::Array{Float64,1},
    param::Array{Float64,2}, c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        # Sparsity pattern of lower-triangular Hessian.
        #     1   2   3
        #    -----------
        # 1 | x
        # 2 | x   x
        # 3 | x   x   x
        #    -----------
        if mode == :Structure
            nz = 0
            for j=1:3
                for i=j:3
                    nz += 1
                    rows[nz] = i
                    cols[nz] = j
                end
            end
        else
            nz = 0

            # dp_{t,I}*p_{t,I}
            nz += 1
            values[nz] = scale*(2*c2*(baseMVA)^2 + param[3,I] + param[8,I])

            # dp_{t,I}*phat_{t-1,I}
            nz += 1
            values[nz] = scale*(-param[8,I])

            # dp_{t,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(-param[8,I])

            # dphat_{t-1,I}*phat_{t-1,I}
            nz += 1
            values[nz] = scale*(param[4,I] + param[8,I])

            # dphat_{t-1,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(param[8,I])

            # ds_{t,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(param[8,I])
        end
    end
    return
end

function eval_f_generator_loose_kernel_cpu(
    I::Int, x::Array{Float64,1}, param::Array{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    f = 0.0

    @inbounds begin
        # x[1]      : p_{t,I}
        # x[2]      : phat_{t-1,I}
        # x[3]      : s_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{ptilde_{t,I}}
        # param[3,I]: lambda_{phat_{t-1,I}}
        # param[4,I]: rho_{p_{t,I}}
        # param[5,I]: rho_{ptilde_{t,I}}
        # param[6,I]: rho_{phat_{t-1,I}}
        # param[7,I]: pbar_{t,I} - z_{p_{t,I}}
        # param[8,I]: ptilde_{t,I} - z_{ptilde_{t,I}}
        # param[9,I]: ptilde_{t-1,I} - z_{phat_{t-1,I}}
        # param[10,I]: mu
        # param[11,I]: xi

        f += c2*(x[1]*baseMVA)^2 + c1*(x[1]*baseMVA) + c0
        f += param[1,I]*(x[1] - param[7,I]) + (0.5*param[4,I])*(x[1] - param[7,I])^2
        f += param[2,I]*(x[1] - param[8,I]) + (0.5*param[5,I])*(x[1] - param[8,I])^2
        f += param[3,I]*(x[2] - param[9,I]) + (0.5*param[6,I])*(x[2] - param[9,I])^2
        f += param[10,I]*(x[1] - x[2] - x[3]) + (0.5*param[11,I])*(x[1] - x[2] - x[3])^2
    end

    f *= scale
    return f
end

function eval_grad_f_generator_loose_kernel_cpu(
    I::Int, x::Array{Float64,1}, g::Array{Float64,1}, param::Array{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        g[1] = 2*c2*(baseMVA)^2*x[1] + c1*baseMVA
        g[1] += param[1,I] + param[4,I]*(x[1] - param[7,I])
        g[1] += param[2,I] + param[5,I]*(x[1] - param[8,I])
        g[1] += param[10,I] + param[11,I]*(x[1] - x[2] - x[3])
        g[1] *= scale
        g[2] = param[3,I] + param[6,I]*(x[2] - param[9,I])
        g[2] += -(param[10,I] + param[11,I]*(x[1] - x[2] - x[3]))
        g[2] *= scale
        g[3] = -(param[10,I] + param[11,I]*(x[1] - x[2] - x[3]))
        g[3] *= scale
    end
    return
end

function eval_h_generator_loose_kernel_cpu(
    I::Int, x::Array{Float64,1}, mode::Symbol, scale::Float64,
    rows, cols, lambda::Array{Float64,1}, values::Array{Float64,1},
    param::Array{Float64,2}, c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        # Sparsity pattern of lower-triangular Hessian.
        #     1   2   3
        #    -----------
        # 1 | x
        # 2 | x   x
        # 3 | x   x   x
        #    -----------
        if mode == :Structure
            nz = 0
            for j=1:3
                for i=j:3
                    nz += 1
                    rows[nz] = i
                    cols[nz] = j
                end
            end
        else
            nz = 0

            # dp_{t,I}*p_{t,I}
            nz += 1
            values[nz] = scale*(2*c2*(baseMVA)^2 + param[4,I] + param[5,I] + param[11,I])

            # dp_{t,I}*phat_{t-1,I}
            nz += 1
            values[nz] = scale*(-param[11,I])

            # dp_{t,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(-param[11,I])

            # dphat_{t-1,I}*phat_{t-1,I}
            nz += 1
            values[nz] = scale*(param[6,I] + param[11,I])

            # dphat_{t-1,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(param[11,I])

            # ds_{t,I}*s_{t,I}
            nz += 1
            values[nz] = scale*(param[11,I])
        end
    end
    return
end