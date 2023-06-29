using Test
using LinearAlgebra
using Printf

using ExaAdmm
using LazyArtifacts

# Data
const INSTANCES_DIR = joinpath(artifact"ExaData", "ExaData")
const MP_DEMAND_DIR = joinpath(INSTANCES_DIR, "mp_demand")


init_time = time()

@testset "Testing ExaAdmm" begin
    @testset "Testing ADMM algorithms on CPUs" begin
        include("algorithms/acopf_update_cpu.jl")
        include("algorithms/mpacopf_update_cpu.jl")
        include("algorithms/qpsub_update_cpu.jl")
    end

    using CUDA
    if CUDA.functional()
        @testset "Testing ADMM algorithms using CUDA.jl" begin
            include("algorithms/acopf_update_gpu.jl")
            include("algorithms/mpacopf_update_gpu.jl")
            include("algorithms/qpsub_update_gpu.jl")
        end
    end

    @testset "Testing ADMM algorithms using KA" begin
        include("algorithms/acopf_update_ka.jl")
        include("algorithms/qpsub_update_ka.jl")
    end
end

println("\nTotal Running Time: $(round(time() - init_time; digits=1)) seconds.")
