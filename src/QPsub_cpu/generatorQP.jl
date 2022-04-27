## individual param
function generatorQP(
    baseMVA::Float64, ngen::Int64,
    curr_lambda_pg::Array{Float64,1},curr_genQP_pg::Array{Float64,1}, curr_busQP_pg::Array{Float64,1}, rho_pg::Array{Float64,1},
    curr_lambda_qg::Array{Float64,1},curr_genQP_qg::Array{Float64,1}, curr_busQP_qg::Array{Float64,1}, rho_qg::Array{Float64,1},
    pgmin_curr::Array{Float64,1}, pgmax_curr::Array{Float64,1},
    qgmin_curr::Array{Float64,1}, qgmax_curr::Array{Float64,1},
    c2::Array{Float64,1}, c1::Array{Float64,1}
)
    for I=1:ngen
        #compute curr_genQP_pg
        curr_genQP_pg[I] = max(pgmin_curr[I],
                        min(pgmax_curr[I],
                            (-(c1[I]*baseMVA + curr_lambda_pg[I] + rho_pg[I]*(-curr_busQP_pg[I]))) / (2*c2[I]*(baseMVA^2) + rho_pg[I])))
        #compute curr_genQP_qg
        curr_genQP_qg[I] = max(qgmin_curr[I],
        min(qgmax_curr[I],
            (-(c1[I]*baseMVA + curr_lambda_qg[I] + rho_qg[I]*(-curr_busQP_qg[I]))) / (2*c2[I]*(baseMVA^2) + rho_qg[I])))
  
    end
    return #curr_sol_genQP
end


## bundled param (require individual param)
function generatorQP(
    env::AdmmEnvSQP,
    sol_genQP::SolutionQP_gen, sol_ACOPF::SolutionACOPF, lam_rho::lam_rho_gen, sol_busQP::SolutionQP_bus
)
    ngen = size(env.data.generators,1)
    baseMVA = env.data.baseMVA
    c2=zeros(Float64,ngen)
    c1=zeros(Float64,ngen)
    pgmin_curr=zeros(Float64,3)
    pgmax_curr=zeros(Float64,3)
    qgmin_curr=zeros(Float64,3)
    qgmax_curr=zeros(Float64,3)

    for i = 1:ngen
        c2[i]=env.data.generators[i].coeff[1]
        c1[i]=env.data.generators[i].coeff[2]
        pgmin_curr[i]=max(env.data.generators[i].Pmin - sol_ACOPF.pg[i], -env.params.trust_rad)
        pgmax_curr[i]=min(env.data.generators[i].Pmax - sol_ACOPF.pg[i], env.params.trust_rad)
        qgmin_curr[i]=max(env.data.generators[i].Qmin - sol_ACOPF.qg[i], -env.params.trust_rad)
        qgmax_curr[i]=min(env.data.generators[i].Qmax - sol_ACOPF.qg[i], env.params.trust_rad)
    end

    tcpu = @timed generatorQP(baseMVA,ngen,lam_rho.lam_pg,sol_busQP)
    return tcpu
end