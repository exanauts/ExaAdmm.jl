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

@kernel function norm_kernel(
    ::Val{n},
    x,
    y
) where {n}
    I = @index(Global, Linear)
    @synchronize
    v = 0.0
    for i in 1:n
        @inbounds v += x[i]*x[i]
    end

    @synchronize
    if I == 1
        y[1] = sqrt(v)
    end
end

function LinearAlgebra.norm(x::Array, device::Nothing)
    return norm(x)
end

function LinearAlgebra.norm(x::CuArray, device::Nothing)
    return CUDA.norm(x)
end

function LinearAlgebra.norm(x, device)
    y = adapt(device, zeros(eltype(x), 1))
    n = length(x)
    norm_kernel(device)(Val{n}(), x, y, ndrange=n)
    KA.synchronize(device)
    ret = y |> Array
    return ret[1]
end
