@inline function eval_f_generator_kernel(
    I::Int, scale::Float64, x::CuDeviceArray{Float64,1}, param::CuDeviceArray{Float64,2},
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
    CUDA.sync_threads()

    return f
end

@inline function eval_grad_f_generator_kernel(
    I::Int, scale::Float64, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, param::CuDeviceArray{Float64,2},
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    tx = threadIdx().x
    @inbounds begin
        if tx == 1
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
    end
    CUDA.sync_threads()
    return
end

@inline function eval_h_generator_kernel(
    I::Int, scale::Float64, x::CuDeviceArray{Float64,1}, A::CuDeviceArray{Float64,2},
    param::CuDeviceArray{Float64,2},
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        A[1,1] = scale*(2*c2*(baseMVA)^2 + param[3,I] + param[8,I])
        A[2,1] = scale*(-param[8,I])
        A[3,1] = scale*(-param[8,I])

        A[1,2] = scale*(-param[8,I])
        A[2,2] = scale*(param[4,I] + param[8,I])
        A[3,2] = scale*(param[8,I])

        A[1,3] = scale*(-param[8,I])
        A[2,3] = scale*(param[8,I])
        A[3,3] = scale*(param[8,I])
    end
    CUDA.sync_threads()
    return
end