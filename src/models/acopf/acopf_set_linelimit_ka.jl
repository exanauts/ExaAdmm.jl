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
    nblk_line = div(mod.nline-1, 64)+1
    ev = set_rateA_kernel_ka(device,64, 64*nblk_line)(mod.nline, mod.membuf, mod.rateA)
    KA.synchronize(device)
end
