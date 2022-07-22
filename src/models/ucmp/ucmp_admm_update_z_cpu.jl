"""
Update variable `z`, representing the artificial variables that are driven to zero in the two-level ADMM.
"""
function admm_update_z(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    return
end
