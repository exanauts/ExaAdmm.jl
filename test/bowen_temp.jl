#test     
#tr = ExaAdmm.build_QP(A,b,l,u)
#ExaAdmm.ExaTron.solveProblem(tr)
#fieldnames(typeof(tr))
#keys(tr.options)
#tr.options["matrix_type"]

function solve_QP_Ipopt(A::Matrix{Float64}, b::Array{Float64, 1}, l::Array{Float64, 1}, u::Array{Float64, 1})
    # @assert size(A,1)==size(A,2)==length(b)==length(l)==length(u)
    # @assert l<=u
    # @assert issymmetric(A) #symmetric #! symmetric check might fail due to numerical residual
    
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

# function trontest1() ## test for build_problem
# @testset "PosDef QP" begin
#     n = 1000
#     obj1 = -193.05853878066543
#     prob = build_problem(; n=n)

#     @testset "Problem definition" begin
#         # @test isa(prob, ExaTron.ExaTronProblem) #check type
#         @test length(prob.x) == length(prob.x_l) == length(prob.x_u) == n
#         @test prob.status == :NotSolved
#     end

#     @testset "Tron: Julia" begin
#         prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
#         ExaTron.solveProblem(prob)
#         println("gap =",abs(prob.f - obj1))
#         # @test_broken prob.f ≈ obj♯ atol=1e-8
#         @test prob.f ≈ obj1 atol=1e-8
#     end

#     # if ExaTron.has_c_library()
#     #     @testset "Tron: Fortran" begin
#     #         ExaTron.setOption(prob, "tron_code", :Fortran)
#     #         prob.x .= 0.5 .* (prob.x_l .+ prob.x_u)
#     #         ExaTron.solveProblem(prob)
#     #         @test prob.f ≈ obj♯ atol=1e-8
#     #     end
#     # end
# end
# end


function trontest2(;n=8) ##test for build_QP and solve_QP_Ipopt
    A=rand(n,n)
    A=A+transpose(A) #symmetric 
    b=rand(n)
    u=rand(n)
    l=u.-10
    
    time_ds = @timed begin
    tr_ds= build_QP_DS(A,b,l,u)
    status_tron_ds = ExaTron.solveProblem(tr_ds)
    end
    
    time_sp = @timed begin
    tr_sp = build_QP_SP(A,b,l,u)
    status_tron_sp = ExaTron.solveProblem(tr_sp)
    end

    time_ip = @timed val,obj,status=solve_QP_Ipopt(A,b,l,u)

    println("time tron dense = ", time_ds.time)
    println("time tron sparse = ", time_sp.time)
    println("time ipopt = ", time_ip.time)
    

    ## this test easily fails when n is large 
    @testset "QP sol: Tron (sparse) vs Ipopt" begin
    #   @test status_tron == :Solve_Succeeded
      @test norm(val - tr_sp.x)≈0 atol=1e-3
      @test obj≈tr_sp.f atol=1e-3 
    #   @test norm(val - tr.x)≈0 rtol=1e-6
    #   @test obj≈tr.f rtol=1e-6 
    end

    @testset "QP sol: Tron (dense) vs Ipopt" begin
        #   @test status_tron == :Solve_Succeeded
          @test norm(val - tr_ds.x)≈0 atol=1e-3
          @test obj≈tr_ds.f atol=1e-3 
        #   @test norm(val - tr.x)≈0 rtol=1e-6
        #   @test obj≈tr.f rtol=1e-6 
        end

    return val, obj, status, tr_ds, status_tron_ds, tr_sp, status_tron_sp
end
#one_line_test 
#val, obj, status, tr_ds, status_tron_ds, tr_sp, status_tron_sp = ExaAdmm.trontest2(n=5)

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