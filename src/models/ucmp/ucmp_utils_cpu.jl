function ucmp_update_uc_membuf_with_ramping_kernel(
    t::Int, ngen::Int,
    vr_u::Array{Float64,1}, vr_z::Array{Float64,1},
    vr_l::Array{Float64,1}, vr_rho::Array{Float64,1},
    uc_membuf::Array{Float64,2}
)
    @inbounds for I=1:ngen
        vh_pz = vr_u[I] + vr_z[I]
        uc_membuf[I,t-1] = vr_rho[I]/2 - vr_rho[I]*vh_pz - vr_l[I]
    end
end