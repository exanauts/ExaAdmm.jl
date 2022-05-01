using Test
using LazyArtifacts
using LinearAlgebra
using Printf
using CUDA

using ExaAdmm

# Data
const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
const MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "mp_demand")

init_time = time()


@testset "Testing ADMM algorithms on CPUs" begin
    include("algorithms/acopf_update_cpu.jl")
    include("algorithms/mpacopf_update_cpu.jl")
end


try
    using CUDA
    if CUDA.functional()
        @testset "Testing ADMM algorithms on GPUs" begin
            include("algorithms/acopf_update_gpu.jl")
            include("algorithms/mpacopf_update_gpu.jl")
        end
    end
catch e
    println("GPU tests have failed: ", e)
end

println("\nTotal Running Time: $(round(time() - init_time; digits=1)) seconds.")
