function ucmp_update_uc_membuf_with_ramping_kernel(
    t::Int, ngen::Int,
    r_u::Array{Float64,1}, r_z::Array{Float64,1},
    r_l::Array{Float64,1}, r_rho::Array{Float64,1},
    uc_membuf::Array{Float64,2}
)
    @inbounds for I=1:ngen
        vh_pz = r_u[2*I] + r_z[2*I]
        uc_membuf[I,t-1] = r_rho[4*I-2]/2 - r_rho[4*I-2]*vh_pz - r_l[4*I-2]
    end
end