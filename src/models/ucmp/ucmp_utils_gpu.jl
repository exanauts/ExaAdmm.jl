function ucmp_update_uc_membuf_with_ramping_kernel(
    t::Int, ngen::Int,
    vr_u::CuDeviceArray{Float64,1}, vr_z::CuDeviceArray{Float64,1},
    vr_l::CuDeviceArray{Float64,1}, vr_rho::CuDeviceArray{Float64,1},
    uc_membuf::CuDeviceArray{Float64,2}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    if I <= ngen
        vh_pz = vr_u[I] + vr_z[I]
        uc_membuf[I,t-1] = vr_rho[I]/2 - vr_rho[I]*vh_pz - vr_l[I]
    end
    return
end