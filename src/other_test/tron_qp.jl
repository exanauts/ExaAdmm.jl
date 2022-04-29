## testing for syntax for Tron 

## example 
# tron = ExaTron.createProblem(n, xl, xu, nele_hess, eval_f_cb, eval_g_cb, eval_h_cb;
# :tol => gtol, :matrix_type => :Dense, :max_minor => 200,
# :frtol => 1e-12)
# tron.x .= x
# it = 0
# avg_tron_minor = 0
# terminate = false

# while !terminate
# it += 1

# # Solve the branch problem.
# status = ExaTron.solveProblem(tron)
# x .= tron.x
# avg_tron_minor += tron.minor_iter


## random problem with diagonal hessian 
function build_problem(; n=10)
    Random.seed!(1)
    # m = 0
    P = sparse(Diagonal(rand(n)) + 2.0 * sparse(I, n, n))
    q = randn(n)
    u =   1. * rand(n)
    l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P)

    eval_f(x) = 0.5 * dot(x, P, x) + dot(q, x)

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, P, x)
        g .+= q
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for i in 1:nnz(P)
                rows[i] = Iz[i]
                cols[i] = Jz[i]
            end
        else
            copy!(values, vals)
        end
    end

    return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12) #with param 
    # return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h) #param
end

function build_QP(A::Matrix{Float64}, b::Array{Float64, 1}, l::Array{Float64, 1}, u::Array{Float64, 1})
    # Random.seed!(1)
    # m = 0

    @assert size(A,1)==size(A,2)==length(b)==length(l)==length(u)
    @assert l<=u
    @assert issymmetric(A) #symmetric
    
    n=length(b)
    P = sparse(A) #make A sparse 
    q = b
    # u =   1. * rand(n)
    # l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P)

    eval_f(x) = 0.5 * dot(x, P, x) + dot(q, x) #obj watch out for 1/2 

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, P, x)
        g .+= q
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for i in 1:nnz(P)
                rows[i] = Iz[i]
                cols[i] = Jz[i]
            end
        else
            copy!(values, vals)
        end
    end

    # return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12) #with param 
    return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h) #original without param
end

#test     
#tr = ExaAdmm.build_QP(A,b,l,u)
#ExaAdmm.ExaTron.solveProblem(tr)
#fieldnames(typeof(tr))
#keys(tr.options)
#tr.options["matrix_type"]

function solve_QP_Ipopt(A::Matrix{Float64}, b::Array{Float64, 1}, l::Array{Float64, 1}, u::Array{Float64, 1})
    @assert size(A,1)==size(A,2)==length(b)==length(l)==length(u)
    @assert l<=u
    @assert issymmetric(A) #symmetric
    
    n=length(b)

    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, l[i]<= x[i=1:n] <=u[i])
    @objective(model, Min, 0.5 * dot(x, A, x) + dot(b, x))
    optimize!(model)
    return JuMP.value.(x), objective_value(model), termination_status(model)
end
#test
#val,obj,status = ExaAdmm.solve_QP_Ipopt(A,b,l,u)

function trontest1() ##
@testset "PosDef QP" begin
    n = 1000
    obj1 = -193.05853878066543
    prob = build_problem(; n=n)

    @testset "Problem definition" begin
        # @test isa(prob, ExaTron.ExaTronProblem) #check type
        @test length(prob.x) == length(prob.x_l) == length(prob.x_u) == n
        @test prob.status == :NotSolved
    end

    @testset "Tron: Julia" begin
        prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
        ExaTron.solveProblem(prob)
        println("gap =",abs(prob.f - obj1))
        # @test_broken prob.f ≈ obj♯ atol=1e-8
        @test prob.f ≈ obj1 atol=1e-8
    end

    # if ExaTron.has_c_library()
    #     @testset "Tron: Fortran" begin
    #         ExaTron.setOption(prob, "tron_code", :Fortran)
    #         prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
    #         ExaTron.solveProblem(prob)
    #         @test prob.f ≈ obj♯ atol=1e-8
    #     end
    # end
end
end

function trontest2(;n=2)
    A=rand(n,n)
    A=A+transpose(A)
    b=rand(n)
    u=rand(n)
    l=u.-10

    tr= build_QP(A,b,l,u)
    status_tron = ExaTron.solveProblem(tr)
    val,obj,status=solve_QP_Ipopt(A,b,l,u)
    
    @testset "QP sol: Tron vs Ipopt" begin
    #   @test status_tron == :Solve_Succeeded
      @test norm(val - tr.x)≈0 atol=1e-6
      @test obj≈tr.f atol=1e-6 
    end

    return tr, val, obj, status, status_tron
end
#test 
#tr,val,obj,status,status_tron = ExaAdmm.trontest2(n=5)

# function trontest2()
# @testset "1-d QP" begin
#     # Solve a simple QP problem: min 0.5*(x-1)^2 s.t. 0 <= x <= 2.0
#     qp_eval_f_cb(x) = 0.5*(x[1]-1)^2
#     function qp_eval_grad_f_cb(x, grad_f)
#         grad_f[1] = x[1] - 1
#     end
#     function qp_eval_h_cb(x, mode, rows, cols, obj_factor, lambda, values)
#         if mode == :Structure
#             rows[1] = 1
#             cols[1] = 1
#         else
#             values[1] = 1.0
#         end
#     end

#     x_l = zeros(1)
#     x_u = zeros(1)
#     x_u[1] = 2.0
#     obj = 0.0

#     prob = ExaTron.createProblem(1, x_l, x_u, 1, qp_eval_f_cb, qp_eval_grad_f_cb, qp_eval_h_cb)
#     @testset "Tron: Julia" begin
#         ExaTron.solveProblem(prob)
#         @test prob.f == obj
#         @test prob.x[1] == 1.0
#     end

#     # if ExaTron.has_c_library()
#     #     @testset "Tron: Fortran" begin
#     #         ExaTron.setOption(prob, "tron_code", :Fortran)
#     #         ExaTron.solveProblem(prob)
#     #         @test prob.f == obj
#     #         @test prob.x[1] == 1.0
#     #     end
#     # end
# end
# end