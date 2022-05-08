function copy_data_kernel(n::Int, dest::CuDeviceArray{Float64,1}, src::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        dest[tx] = src[tx]
    end
    return
end

function vector_difference(n::Int, c, a, b)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        c[tx] = a[tx] - b[tx]
    end

    return
end
