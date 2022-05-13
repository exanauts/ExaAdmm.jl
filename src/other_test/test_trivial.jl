# function test_trivial(
# )
#     @printf(" ** test module update ** ")
#     return
# end

using LinearAlgebra
using SparseArrays

A=rand(3,3)
B=A+transpose(A)
C=sparse(B)

findnz(C)
nnz(C)
issymmetric(C)