## individual param
function branchQP(
    nline::Int,
    curr_lambda_brQP,curr_lambda_busQP, rho, curr_sol_acopf, curr_sol_busQP, curr_pi_acopf
)
    for I=1:nline
        #compute curr_sol_brQP
    end

    return #curr_sol_brQP
end


## bundled param (require individual param)
# function generatorQP(
#     model::Model{Float64,Array{Float64,1},Array{Int,1}},
#     baseMVA::Float64, u, xbar, zu, lu, rho_u
# )
#     tcpu = @timed generator_kernel_two_level(.....)
#     return tcpu
# end