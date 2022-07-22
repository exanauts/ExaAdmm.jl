"""
Update variable `x`, representing the variables for generators and lines in the component-based decomposition of ACOPF.
"""
function admm_update_x(
    env::AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::UCMPModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    device::Nothing=nothing
)
    return
end