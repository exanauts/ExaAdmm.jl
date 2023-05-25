@kernel function set_rateA_kernel_ka(nline::Int, param, rateA)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= nline
        @inbounds begin
            param[29,tx] = (rateA[tx] == 0.0) ? 1e3 : rateA[tx]
        end
    end
end

function acopf_set_linelimit(
    env::AdmmEnv,
    mod::AbstractOPFModel,
    info::IterationInformation,
    device
)
    ev = set_rateA_kernel_ka(device,64,mod.nline)(mod.nline, mod.membuf, mod.rateA)
    KA.synchronize(device)
end
