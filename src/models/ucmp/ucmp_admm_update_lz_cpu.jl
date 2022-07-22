"""
Compute and update multipliers `λ_z` for the augmented Lagrangian wit respect to `z=0` constraint.
"""
function admm_update_lz(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    return
end