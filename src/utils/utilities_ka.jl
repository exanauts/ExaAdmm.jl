@kernel function copy_data_kernel_ka(n::Int, dest, src)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        dest[tx] = src[tx]
    end
end

@kernel function vector_difference_ka(n::Int, c, a, b)
    I = @index(Group, Linear)
    J = @index(Local, Linear)
    tx = J + (@groupsize()[1] * (I - 1))

    if tx <= n
        c[tx] = a[tx] - b[tx]
    end
end
@kernel function norm_kernel(::Val{n}, x,
                    y
                    ) where {n}
    tx = @index(Local, Linear)
    bx = @index(Group, Linear)

    _x = @localmem Float64 (n,)
    _x[tx] = x[tx]
    @synchronize

    v = ExaTron.dnrm2(n, _x, 1, tx)
    if bx == 1 && tx == 1
        y[1] = v
    end
    @synchronize

end
function LinearAlgebra.norm(x, device)
    y = KAArray{Float64}(1, device)
    n = length(x)
    wait(norm_kernel(device,n)(Val{n}(), x, y,ndrange=(n,),dependencies=Event(device)))
    ret = y |> Array
    return ret[1]
end
