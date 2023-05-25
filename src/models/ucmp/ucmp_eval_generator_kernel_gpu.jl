#########################################################
# Table of variables and parameters
        # l: lambda; r_l: lambda for ramp variables; 
        #
        # x[1]      : p_{t,I}
        # x[2]      : q_{t,I}
        # x[3]      : phat_{t,I}
        # x[4]      : v_{t,I}
        # x[5]      : w_{t,I}
        # x[6]      : y_{t,I}
        # x[7]      : vhat_{t,I}
        # x[8]      : s^U_{t,I}
        # x[9]      : s^D_{t,I}
        # x[10]     : s^{p,UB}_{t,I}
        # x[11]     : s^{p,LB}_{t,I}
        # x[12]     : s^{q,UB}_{t,I}
        # x[13]     : s^{q,LB}_{t,I}
        # param[1,I]: lambda_{p_{t,I}}
        # param[2,I]: lambda_{q_{t,I}}
        # param[3,I]: lambda_{phat_{t,I}}
        # param[4,I]: lambda_{v_{t,I}}
        # param[5,I]: lambda_{w_{t,I}}
        # param[6,I]: lambda_{y_{t,I}}
        # param[7,I]: lambda_{vhat_{t,I}}
        # param[8,I]: lambda_{su_{t,I}}
        # param[9,I]: lambda_{sd_{t,I}}
        # param[10,I]: lambda_{pUB_{t,I}}
        # param[11,I]: lambda_{pLB_{t,I}}
        # param[12,I]: lambda_{qUB_{t,I}}
        # param[13,I]: lambda_{qLB_{t,I}}
        # param[14,I]: rho_{p_{t,I}}
        # param[15,I]: rho_{q_{t,I}}
        # param[16,I]: rho_{phat_{t,I}}
        # param[17,I]: rho_{v_{t,I}}
        # param[18,I]: rho_{w_{t,I}}
        # param[19,I]: rho_{y_{t,I}}
        # param[20,I]: rho_{vhat_{t,I}}
        # param[21,I]: rho_{su_{t,I}}
        # param[22,I]: rho_{sd_{t,I}}
        # param[23,I]: rho_{pUB_{t,I}}
        # param[24,I]: rho_{pLB_{t,I}}
        # param[25,I]: rho_{qUB_{t,I}}
        # param[26,I]: rho_{qLB_{t,I}}
        # param[27,I]: p_{t,I(i)} - z_{p_{t,I}}
        # param[28,I]: q_{t,I(i)} - z_{q_{t,I}}
        # param[29,I]: p_{t-1,I(i)} - z_{phat_{t-1,I}}
        # param[30,I]: vbar_{t,I} - z_{v_{t,I}}
        # param[31,I]: wbar_{t,I} - z_{w_{t,I}}
        # param[32,I]: ybar_{t,I} - z_{y_{t,I}}
        # param[33,I]: vbar_{t-1,I} - z_{vhat_{t,I}}
        # param[34,I]: RUg
        # param[35,I]: SUg
        # param[36,I]: RDg
        # param[37,I]: SDg
        # param[38,I]: pmax
        # param[39,I]: pmin
        # param[40,I]: qmax
        # param[41,I]: qmin
        # param[42,I]: mu
        # param[43,I]: xi
#########################################################

# NOTE: I cannot be defined as type Int because it is not Int type

@inline function eval_f_generator_continuous_kernel_gpu(
    t::Int, I, x::CuDeviceArray{Float64,1}, param::CuDeviceArray{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    f = 0.0

    @inbounds begin
        f += c2*(x[1]*baseMVA)^2 + c1*(x[1]*baseMVA) + c0
        f += param[1,I]*(x[1] - param[27,I]) + (0.5*param[14,I])*(x[1] - param[27,I])^2
        f += param[2,I]*(x[2] - param[28,I]) + (0.5*param[15,I])*(x[2] - param[28,I])^2
        f += param[4,I]*(x[4] - param[30,I]) + (0.5*param[17,I])*(x[4] - param[30,I])^2
        f += param[5,I]*(x[5] - param[31,I]) + (0.5*param[18,I])*(x[5] - param[31,I])^2
        f += param[6,I]*(x[6] - param[32,I]) + (0.5*param[19,I])*(x[6] - param[32,I])^2
        f += param[10,I]*(x[1] - param[38,I]*x[4] - x[10]) + (0.5*param[23,I])*(x[1] - param[38,I]*x[4] - x[10])^2
        f += param[11,I]*(x[1] - param[39,I]*x[4] + x[11]) + (0.5*param[24,I])*(x[1] - param[39,I]*x[4] + x[11])^2
        f += param[12,I]*(x[2] - param[40,I]*x[4] - x[12]) + (0.5*param[25,I])*(x[2] - param[40,I]*x[4] - x[12])^2
        f += param[13,I]*(x[2] - param[41,I]*x[4] + x[13]) + (0.5*param[26,I])*(x[2] - param[41,I]*x[4] + x[13])^2

        if t > 1
            f += param[3,I]*(x[3] - param[29,I]) + (0.5*param[16,I])*(x[3] - param[29,I])^2
            f += param[7,I]*(x[7] - param[33,I]) + (0.5*param[20,I])*(x[7] - param[33,I])^2
            f += param[8,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
            f += (0.5*param[21,I])*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])^2
            f += param[9,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
            f += (0.5*param[22,I])*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])^2
        end
    end

    f *= scale
    CUDA.sync_threads()

    return f
end

@inline function eval_grad_f_generator_continuous_kernel_gpu(
    t::Int, I, x::CuDeviceArray{Float64,1}, g::CuDeviceArray{Float64,1}, param::CuDeviceArray{Float64,2}, scale::Float64,
    c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    # tx = threadIdx().x
    # if tx == 1
        @inbounds begin
            g[1] = 2*c2*(baseMVA)^2*x[1] + c1*baseMVA
            g[1] += param[1,I] + param[14,I]*(x[1] - param[27,I])
            g[1] += param[10,I] + param[23,I]*(x[1] - param[38,I]*x[4] - x[10])
            g[1] += param[11,I] + param[24,I]*(x[1] - param[39,I]*x[4] + x[11])
            g[2] = param[2,I] + param[15,I]*(x[2] - param[28,I])
            g[2] += param[12,I] + param[25,I]*(x[2] - param[40,I]*x[4] - x[12])
            g[2] += param[13,I] + param[26,I]*(x[2] - param[41,I]*x[4] + x[13])
            g[3] = 0
            g[4] = param[4,I] + param[17,I]*(x[4] - param[30,I])
            g[4] += -param[10,I]*param[38,I] - param[23,I]*param[38,I]*(x[1] - param[38,I]*x[4] - x[10])
            g[4] += -param[11,I]*param[39,I] - param[24,I]*param[39,I]*(x[1] - param[39,I]*x[4] + x[11])
            g[4] += -param[12,I]*param[40,I] - param[25,I]*param[40,I]*(x[2] - param[40,I]*x[4] - x[12])
            g[4] += -param[13,I]*param[41,I] - param[26,I]*param[41,I]*(x[2] - param[41,I]*x[4] + x[13])
            g[5] = param[5,I] + param[18,I]*(x[5] - param[31,I])
            g[6] = param[6,I] + param[19,I]*(x[6] - param[32,I])
            g[7] = 0
            g[8] = 0
            g[9] = 0
            g[10] = -param[10,I] - param[23,I]*(x[1] - param[38,I]*x[4] - x[10])
            g[11] = param[11,I] + param[24,I]*(x[1] - param[39,I]*x[4] + x[11])
            g[12] = -param[12,I] - param[25,I]*(x[2] - param[40,I]*x[4] - x[12])
            g[13] = param[13,I] + param[26,I]*(x[2] - param[41,I]*x[4] + x[13])

            if t > 1
                g[1] += param[8,I] + param[21,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
                g[1] += param[9,I] + param[22,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
                g[3] += param[3,I] + param[16,I]*(x[3] - param[29,I])
                g[3] += -param[8,I] - param[21,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
                g[3] += -param[9,I] - param[22,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
                g[4] += param[9,I]*param[36,I] + param[22,I]*param[36,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
                g[5] += -param[8,I]*param[35,I] - param[21,I]*param[35,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
                g[6] += param[9,I]*param[37,I] + param[22,I]*param[37,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
                g[7] += param[7,I] + param[20,I]*(x[7] - param[33,I])
                g[7] += -param[8,I]*param[34,I] - param[21,I]*param[34,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
                g[8] += -param[8,I] - param[21,I]*(x[1] - x[3] - param[34,I]*x[7] - param[35,I]*x[5] - x[8])
                g[9] += param[9,I] + param[22,I]*(x[1] - x[3] + param[36,I]*x[4] + param[37,I]*x[6] + x[9])
            end

            g[1] *= scale
            g[2] *= scale
            g[3] *= scale
            g[4] *= scale
            g[5] *= scale
            g[6] *= scale
            g[7] *= scale
            g[8] *= scale
            g[9] *= scale
            g[10] *= scale
            g[11] *= scale
            g[12] *= scale
            g[13] *= scale
        end
    # end
    CUDA.sync_threads()
    return
end

# Nonzero entries of Hessian
# H[1,1] = scale*(2*c2*(baseMVA)^2 + param[14,I] + param[23,I] + param[24,I] + {param[21,I]+param[22,I]} ) *
# H[1,3] = 0 + {scale*(-param[21,I]-param[22,I])}                                                          *
# H[1,4] = scale*(-param[23,I]*param[38,I] - param[24,I]*param[39,I] + {param[22,I]*param[36,I]})          *
# H[1,5] = 0 + {scale*(-param[21,I]*param[35,I])}                                                          *
# H[1,6] = 0 + {scale*(param[22,I]*param[37,I])}                                                           *
# H[1,7] = 0 + {scale*(-param[21,I]*param[34,I])}                                                          *
# H[1,8] = 0 + {scale*(-param[21,I])}                                                                      *
# H[1,9] = 0 + {scale*(param[22,I])}                                                                       *
# H[1,10] = -scale*param[23,I]
# H[1,11] = scale*param[24,I]
# H[2,2] = scale*(param[15,I] + param[25,I] + param[26,I])
# H[2,4] = scale*(-param[25,I]*param[40,I] - param[26,I]*param[41,I])
# H[2,12] = -scale*param[25,I]
# H[2,13] = scale*param[26,I]
# H[3,1] = 0 + {scale*(-param[21,I]-param[22,I])}                                                          *
# H[3,3] = 0 + {scale*(param[16,I] + param[21,I] + param[22,I])}                                           *
# H[3,4] = 0 + {scale*(-param[22,I]*param[36,I])}                                                          *
# H[3,5] = 0 + {scale*param[21,I]*param[35,I]}                                                             *
# H[3,6] = 0 + {scale*(-param[22,I]*param[37,I])}                                                          *
# H[3,7] = 0 + {scale*param[21,I]*param[34,I]}                                                             *
# H[3,8] = 0 + {scale*param[21,I]}                                                                         *
# H[3,9] = 0 + {scale*(-param[22,I])}                                                                      *
# H[4,1] = scale*(-param[23,I]*param[38,I] - param[24,I]*param[39,I] + {param[22,I]*param[36,I]})          *
# H[4,2] = scale*(-param[25,I]*param[40,I] - param[26,I]*param[41,I])
# H[4,3] = 0 + {scale*(-param[22,I]*param[36,I])}                                                          *
# H[4,4] = scale*(param[17,I] + param[23,I]*param[38,I]^2 + param[24,I]*param[39,I]^2 
#                   + param[25,I]*param[40,I]^2 + param[26,I]*param[41,I]^2 + {param[22,I]*param[36,I]^2}) *
# H[4,6] = 0 + {scale*param[22,I]*param[36,I]*param[37,I]}                                                 *
# H[4,9] = 0 + {scale*param[22,I]*param[36,I]}                                                             *
# H[4,10] = scale*(param[23,I]*param[38,I])
# H[4,11] = scale*(-param[24,I]*param[39,I])
# H[4,12] = scale*(param[25,I]*param[40,I])
# H[4,13] = scale*(-param[26,I]*param[41,I])
# H[5,1] = 0 + {scale*(-param[21,I]*param[35,I])}                                                          *
# H[5,3] = 0 + {scale*param[21,I]*param[35,I]}                                                             *
# H[5,5] = scale*(param[18,I] + {param[21,I]*param[35,I]^2})                                               *
# H[5,7] = 0 + {scale*param[21,I]*param[35,I]*param[34,I]}                                                 *
# H[5,8] = 0 + {scale*param[21,I]*param[35,I]}                                                             *
# H[6,1] = 0 + {scale*param[22,I]*param[37,I]}
# H[6,3] = 0 + {scale*(-param[22,I]*param[37,I])}
# H[6,4] = 0 + {scale*param[22,I]*param[37,I]*param[36,I]}
# H[6,6] = scale*(param[19,I] + {param[22,I]*param[37,I]^2}
# H[6,9] = 0 + {scale*param[22,I]*param[37,I]}
# H[7,1] = 0 + {scale*(-param[21,I]*param[34,I])}
# H[7,3] = 0 + {scale*param[21,I]*param[34,I]}
# H[7,5] = 0 + {scale*param[21,I]*param[34,I]*param[35,I]}
# H[7,7] = 0 + {scale*(param[20,I]+param[21,I]*param[34,I]^2)}
# H[7,8] = 0 + {scale*param[21,I]*param[34,I]}
# H[8,1] = 0 + {scale*(-param[21,I])}
# H[8,3] = 0 + {scale*param[21,I]}
# H[8,5] = 0 + {scale*param[21,I]*param[35,I]}
# H[8,7] = 0 + {scale*param[21,I]*param[34,I]}
# H[8,8] = 0 + {scale*param[21,I]}
# H[9,1] = 0 + {scale*param[22,I]}
# H[9,3] = 0 + {scale*(-param[22,I])}
# H[9,4] = 0 + {scale*param[22,I]*param[36,I]}
# H[9,6] = 0 + {scale*param[22,I]*param[37,I]}
# H[9,9] = 0 + {scale*param[22,I]}
# H[10,1] = scale*(-param[23,I])
# H[10,4] = scale*param[23,I]*param[38,I]
# H[10,10] = scale*param[23,I]
# H[11,1] = scale*param[24,I]
# H[11,4] = scale*(-param[24,I]*param[39,I])
# H[11,11] = scale*param[24,I]
# H[12,2] = scale*(-param[25,I])
# H[12,4] = scale*param[25,I]*param[40,I]
# H[12,12] = scale*param[25,I]
# H[13,2] = scale*param[26,I]
# H[13,4] = scale*(-param[26,I]*param[41,I])
# H[13,13] = scale*param[26,I]



@inline function eval_h_generator_continuous_kernel_gpu(
    t::Int, I, x::CuDeviceArray{Float64,1}, H::CuDeviceArray{Float64,2},
    param::CuDeviceArray{Float64,2}, scale::Float64, c2::Float64, c1::Float64, c0::Float64, baseMVA::Float64
)
    @inbounds begin
        for i in 1:13
            for j in 1:13
                H[i,j] = 0
            end
        end
        H[1,1] = scale*(2*c2*(baseMVA)^2 + param[14,I] + param[23,I] + param[24,I] )
        H[2,2] = scale*(param[15,I] + param[25,I] + param[26,I])
        H[4,1] = scale*(-param[23,I]*param[38,I] - param[24,I]*param[39,I])
        H[1,4] = scale*(-param[23,I]*param[38,I] - param[24,I]*param[39,I])
        H[4,2] = scale*(-param[25,I]*param[40,I] - param[26,I]*param[41,I])
        H[2,4] = scale*(-param[25,I]*param[40,I] - param[26,I]*param[41,I])
        H[4,4] = scale*(param[17,I] + param[23,I]*param[38,I]^2 + param[24,I]*param[39,I]^2 
                          + param[25,I]*param[40,I]^2 + param[26,I]*param[41,I]^2)
        H[5,5] = scale*(param[18,I])
        H[6,6] = scale*param[19,I]
        H[10,1] = scale*(-param[23,I])
        H[1,10] = scale*(-param[23,I])
        H[10,4] = scale*param[23,I]*param[38,I]
        H[4,10] = scale*param[23,I]*param[38,I]
        H[10,10] = scale*param[23,I]
        H[11,1] = scale*param[24,I]
        H[1,11] = scale*param[24,I]
        H[11,4] = scale*(-param[24,I]*param[39,I])
        H[4,11] = scale*(-param[24,I]*param[39,I])
        H[11,11] = scale*param[24,I]
        H[12,2] = scale*(-param[25,I])
        H[2,12] = scale*(-param[25,I])
        H[12,4] = scale*param[25,I]*param[40,I]
        H[4,12] = scale*param[25,I]*param[40,I]
        H[12,12] = scale*param[25,I]
        H[13,2] = scale*param[26,I]
        H[2,13] = scale*param[26,I]
        H[13,4] = scale*(-param[26,I]*param[41,I])
        H[4,13] = scale*(-param[26,I]*param[41,I])
        H[13,13] = scale*param[26,I]

        if t > 1
            H[1,1] += scale*(param[21,I]+param[22,I])
            H[3,1] += scale*(-param[21,I]-param[22,I])
            H[1,3] += scale*(-param[21,I]-param[22,I])
            H[3,3] += scale*(param[16,I] + param[21,I] + param[22,I])
            H[4,1] += scale*(param[22,I]*param[36,I])
            H[1,4] += scale*(param[22,I]*param[36,I])
            H[4,3] += scale*(-param[22,I]*param[36,I])
            H[3,4] += scale*(-param[22,I]*param[36,I])
            H[4,4] += scale*(param[22,I]*param[36,I]^2)
            H[5,1] += scale*(-param[21,I]*param[35,I])
            H[1,5] += scale*(-param[21,I]*param[35,I])
            H[5,3] += scale*param[21,I]*param[35,I]
            H[3,5] += scale*param[21,I]*param[35,I]
            H[5,5] += scale*(param[21,I]*param[35,I]^2)
            H[6,1] += scale*(param[22,I]*param[37,I])
            H[1,6] += scale*(param[22,I]*param[37,I])
            H[6,3] += scale*(-param[22,I]*param[37,I])
            H[3,6] += scale*(-param[22,I]*param[37,I])
            H[6,4] += scale*param[22,I]*param[36,I]*param[37,I]
            H[4,6] += scale*param[22,I]*param[36,I]*param[37,I]
            H[6,6] += scale*param[22,I]*param[37,I]^2
            H[7,1] += scale*(-param[21,I]*param[34,I])                                                         
            H[1,7] += scale*(-param[21,I]*param[34,I])                                                         
            H[7,3] += scale*param[21,I]*param[34,I]
            H[3,7] += scale*param[21,I]*param[34,I]
            H[7,5] += scale*param[21,I]*param[35,I]*param[34,I]
            H[5,7] += scale*param[21,I]*param[35,I]*param[34,I]
            H[7,7] += scale*(param[20,I]+param[21,I]*param[34,I]^2)
            H[8,1] += scale*(-param[21,I])
            H[1,8] += scale*(-param[21,I])
            H[8,3] += scale*param[21,I]
            H[3,8] += scale*param[21,I]
            H[8,5] += scale*param[21,I]*param[35,I]
            H[5,8] += scale*param[21,I]*param[35,I]
            H[8,7] += scale*param[21,I]*param[34,I]
            H[7,8] += scale*param[21,I]*param[34,I]
            H[8,8] += scale*param[21,I]
            H[9,1] += scale*(param[22,I])
            H[1,9] += scale*(param[22,I])
            H[9,3] += scale*(-param[22,I])
            H[3,9] += scale*(-param[22,I])
            H[9,4] += scale*param[22,I]*param[36,I]
            H[4,9] += scale*param[22,I]*param[36,I]
            H[9,6] += scale*param[22,I]*param[37,I]
            H[6,9] += scale*param[22,I]*param[37,I]
            H[9,9] += scale*param[22,I]
        end
    end
    CUDA.sync_threads()
    return
end