## individual param
function busQP(
    nbus::Int,
    curr_lambda_genQP,curr_lambda_brQP, rho, curr_sol_genQP
)
    for I=1:nbus
        #compute curr_sol_busQP
    end

    return #curr_sol_busQP
end


## bundled param (require individual param)
# function generatorQP(
#     model::Model{Float64,Array{Float64,1},Array{Int,1}},
#     baseMVA::Float64, u, xbar, zu, lu, rho_u
# )
#     tcpu = @timed generator_kernel_two_level()
#     return tcpu
# end