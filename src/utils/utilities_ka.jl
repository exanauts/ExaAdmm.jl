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
