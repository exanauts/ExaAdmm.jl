"""
    build_QP_*()

- build any box-constrained QP with      
- use Exatron.createproblem()
- TODO: clean up  
"""


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
# function build_problem(; n=10)
#     Random.seed!(1)
#     # m = 0
#     P = sparse(Diagonal(rand(n)) + 2.0 * sparse(I, n, n))
#     q = randn(n)
#     u =   1. * rand(n)
#     l = - 100. * rand(n)
#     Iz, Jz, vals = findnz(P)

#     eval_f(x) = 0.5 * dot(x, P, x) + dot(q, x)

#     function eval_g(x, g)
#         fill!(g, 0)
#         mul!(g, P, x)
#         g .+= q
#     end

#     function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
#         if mode == :Structure
#             for i in 1:nnz(P)
#                 rows[i] = Iz[i]
#                 cols[i] = Jz[i]
#             end
#         else
#             copy!(values, vals)
#         end
#     end

#     return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12) #with param 
#     # return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h) #param
# end

function build_QP_SP(A::Matrix{Float64}, b::Array{Float64, 1}, l::Array{Float64, 1}, u::Array{Float64, 1})
    # Random.seed!(1)
    # m = 0

    # @assert size(A,1)==size(A,2)==length(b)==length(l)==length(u)
    # @assert l<=u
    # @assert issymmetric(A) #symmetric #! symmetric check might fail due to numerical residual
    
    n=length(b)
    P = sparse(A) #make A sparse to use nnz() and findnz()
    q = b
    # u =   1. * rand(n)
    # l = - 100. * rand(n)
    Iz, Jz, vals = findnz(P) #does not see symmetric record n*n

    eval_f(x) = 0.5 * dot(x, P, x) + dot(q, x) #obj watch out for 1/2 

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, P, x)
        g .+= q
    end
    
    #eval_h store all n*n entries
    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            for i in 1:nnz(P) #does not symmetric
                rows[i] = Iz[i]
                cols[i] = Jz[i]
            end
            # for j=1:n
            #     for i in j:n 
            #         nz += 1
            #         rows[nz] = i
            #         cols[nz] = j
            #     end
            # end
        else
            copy!(values, vals)
            # for j=1:n
            #     for i in j:n 
            #         nz += 1
            #         values[nz] = A[i, j]
            #     end
            # end
        end
    end

    return ExaTron.createProblem(n, l, u, nnz(P), eval_f, eval_g, eval_h) #original without param
end


function build_QP_DS(A::Matrix{Float64}, b::Array{Float64, 1}, l::Array{Float64, 1}, u::Array{Float64, 1})
    # Random.seed!(1)
    # m = 0

    # @assert size(A,1)==size(A,2)==length(b)==length(l)==length(u)
    # @assert l<=u
    # @assert issymmetric(A) #! symmetric check might fail due to numerical residual
    
    n=length(b)


    eval_f(x) = 0.5 * dot(x, A, x) + dot(b, x) #obj watch out for 1/2 

    function eval_g(x, g)
        fill!(g, 0)
        mul!(g, A, x)
        g .+= b
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            # for i in 1:nnz(P)
            #     rows[i] = Iz[i]
            #     cols[i] = Jz[i]
            # end
            nz = 1
            for j=1:n
                for i in j:n 
                    rows[nz] = i
                    cols[nz] = j
                    nz += 1
                end
            end
        else
            # copy!(values, vals)
            nz = 1
            for j=1:n
                for i in j:n 
                    values[nz] = A[i, j]
                    nz += 1
                end
            end
        end
    end
    #with Youngdae's Param
    return ExaTron.createProblem(n, l, u, Int64((n+1)*n/2), eval_f, eval_g, eval_h; :tol=> 1e-6, :matrix_type => :Dense, :max_minor => 200, :frtol => 1e-12)  
    # return ExaTron.createProblem(n, l, u, Int64((n+1)*n/2), eval_f, eval_g, eval_h) #original without param
end
