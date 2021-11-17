
using CUDA
using ExaAdmm
using LazyArtifacts

const case = joinpath("matpower", "data", "case_ACTIVSg70k.m")

# warm-up
env, mod = ExaAdmm.solve_acopf(case; rho_pq=3e4, rho_va=3e5, outer_iterlim=2, inner_iterlim=2, scale=1e-5, tight_factor=0.99, use_gpu=true, obj_scale=2.0)
env, mod = ExaAdmm.solve_acopf(case; rho_pq=3e4, rho_va=3e5, outer_iterlim=11, inner_iterlim=1000, scale=1e-5, tight_factor=0.99, use_gpu=true, obj_scale=2.0)
