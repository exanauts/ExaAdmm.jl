using ExaAdmm
using LazyArtifacts

const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
const case = joinpath(INSTANCES_DIR, "ACTIVSg70K.raw")

env, mod = ExaAdmm.solve_acopf(case; rho_pq=1e1, rho_va=1e3, outer_iterlim=20, inner_iterlim=20, scale=1e-4, tight_factor=0.99, use_gpu=true)