using CUDA
using CUDAKernels

function KAArray{T}(n::Int, device::CUDADevice) where {T}
    return CuArray{T}(undef, n)
end
