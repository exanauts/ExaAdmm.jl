"""
Compute and update the primal and dual residuals at current iteration.

### Arguments

- `env::AdmmEnv` -- struct that defines the environment of ADMM
- `mod::UCMPModel` -- struct that defines model

### Notes

The primal and dual residuals are stored in `mod.solution.rp` and `mod.solution.rd`, respectively.
"""
function admm_update_residual(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    return
end
