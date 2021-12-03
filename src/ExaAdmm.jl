module ExaAdmm

using Printf
using FileIO
using DelimitedFiles
using LinearAlgebra
using MPI
using CUDA
using ExaTron

include("common/parse_matpower.jl")
include("common/opfdata.jl")
include("common/environment.jl")
include("common/check_violations.jl")
include("common/acopf_admm.jl")
include("common/acopf_admm_utils.jl")

include("cpu/init_solution_cpu.jl")
include("cpu/generator_kernel_cpu.jl")
include("cpu/eval_linelimit_kernel_cpu.jl")
include("cpu/auglag_linelimit_kernel_cpu.jl")
include("cpu/bus_kernel_cpu.jl")
include("cpu/acopf_admm_update_x_cpu.jl")
include("cpu/acopf_admm_update_xbar_cpu.jl")
include("cpu/acopf_admm_update_z_cpu.jl")
include("cpu/acopf_admm_update_l_cpu.jl")
include("cpu/acopf_admm_update_residual_cpu.jl")
include("cpu/acopf_admm_update_lz_cpu.jl")
include("cpu/acopf_admm_prepoststep_cpu.jl")
include("cpu/acopf_set_linelimit_cpu.jl")

include("gpu/utilities_gpu.jl")
include("gpu/init_solution_gpu.jl")
include("gpu/generator_kernel_gpu.jl")
include("gpu/eval_linelimit_kernel_gpu.jl")
include("gpu/tron_linelimit_kernel.jl")
include("gpu/auglag_linelimit_kernel_gpu.jl")
include("gpu/bus_kernel_gpu.jl")
include("gpu/acopf_admm_update_x_gpu.jl")
include("gpu/acopf_admm_update_xbar_gpu.jl")
include("gpu/acopf_admm_update_z_gpu.jl")
include("gpu/acopf_admm_update_l_gpu.jl")
include("gpu/acopf_admm_update_residual_gpu.jl")
include("gpu/acopf_admm_update_lz_gpu.jl")
include("gpu/acopf_admm_prepoststep_gpu.jl")
include("gpu/acopf_set_linelimit_gpu.jl")

end # module
