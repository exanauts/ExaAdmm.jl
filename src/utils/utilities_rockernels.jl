using AMDGPU
using ROCKernels

function KAArray{T}(n::Int, device::ROCDevice) where {T}
    return ROCArray{T}(undef, n)
end
