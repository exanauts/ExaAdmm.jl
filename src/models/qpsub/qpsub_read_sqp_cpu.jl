## individual param
function readSQP(
    SQP_output, coeff::Coeff_SQP
)
    #coeff = SQP_output
    #assert to check SQP_output make sense

    return 
end


## bundled param (require individual param)
function readSQP(
    SQP_output, mod::ModelQpsub
)
    #mod. coeff = SQP_output
    #assert to check SQP_output make sense
    return 
end

#list of assert
     #@assert Hpg and hpg with c2 and c1 and its diagonal  
     #@assert no Hqg or hqg terms from SQP