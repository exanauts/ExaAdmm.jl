
using CUDA
using ExaAdmm
using LazyArtifacts

const case = joinpath("matpower", "data", "case_ACTIVSg70k.m")

env, mod = ExaAdmm.solve_acopf(case; rho_pq=1e1, rho_va=1e3, outer_iterlim=20, inner_iterlim=20, scale=1e-4, tight_factor=0.99, use_gpu=true)
CUDA.@profile env, mod = ExaAdmm.solve_acopf(case; rho_pq=1e1, rho_va=1e3, outer_iterlim=3, inner_iterlim=20, scale=1e-4, tight_factor=0.99, use_gpu=true)
