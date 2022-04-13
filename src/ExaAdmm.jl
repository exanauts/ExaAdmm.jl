module ExaAdmm

using Printf
using FileIO
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using MPI
using CUDA
using ExaTron
using Random

using JuMP
using Ipopt

include("common/parse_matpower.jl")
include("common/opfdata.jl")
include("common/environment.jl")
include("common/check_violations.jl")
include("common/acopf_admm_utils.jl")
include("common/acopf_admm.jl")
include("common/acopf_admm_increment.jl")
include("common/acopf_ipopt.jl")

# CPU implementation for solving single-period ACOPF
include("cpu/acopf_init_solution_cpu.jl")
include("cpu/acopf_generator_kernel_cpu.jl")
include("cpu/acopf_eval_linelimit_kernel_cpu.jl")
include("cpu/acopf_auglag_linelimit_kernel_cpu.jl")
include("cpu/acopf_bus_kernel_cpu.jl")
include("cpu/acopf_admm_update_x_cpu.jl")
include("cpu/acopf_admm_update_xbar_cpu.jl")
include("cpu/acopf_admm_update_z_cpu.jl")
include("cpu/acopf_admm_update_l_cpu.jl")
include("cpu/acopf_admm_update_residual_cpu.jl")
include("cpu/acopf_admm_update_lz_cpu.jl")
include("cpu/acopf_admm_prepoststep_cpu.jl")

# GPU implementation for solving single-period ACOPF
include("gpu/utilities_gpu.jl")
include("gpu/acopf_init_solution_gpu.jl")
include("gpu/acopf_generator_kernel_gpu.jl")
include("gpu/acopf_eval_linelimit_kernel_gpu.jl")
include("gpu/acopf_tron_linelimit_kernel.jl")
include("gpu/acopf_auglag_linelimit_kernel_gpu.jl")
include("gpu/acopf_bus_kernel_gpu.jl")
include("gpu/acopf_admm_update_x_gpu.jl")
include("gpu/acopf_admm_update_xbar_gpu.jl")
include("gpu/acopf_admm_update_z_gpu.jl")
include("gpu/acopf_admm_update_l_gpu.jl")
include("gpu/acopf_admm_update_residual_gpu.jl")
include("gpu/acopf_admm_update_lz_gpu.jl")
include("gpu/acopf_admm_prepoststep_gpu.jl")

# Multi-period ACOPF
include("common/mpacopf_environment.jl")
include("common/mpacopf_admm_utils.jl")
include("common/mpacopf_check_violations.jl")
include("common/mpacopf_admm.jl")
include("common/mpacopf_admm_increment.jl")

# CPU implementation for solving multi-period ACOPF
include("cpu/mpacopf_init_solution_cpu.jl")
include("cpu/mpacopf_eval_generator_kernel_cpu.jl")
include("cpu/mpacopf_auglag_generator_kernel_cpu.jl")
include("cpu/mpacopf_bus_kernel_cpu.jl")
include("cpu/mpacopf_admm_update_x_cpu.jl")
include("cpu/mpacopf_admm_update_xbar_cpu.jl")
include("cpu/mpacopf_admm_update_z_cpu.jl")
include("cpu/mpacopf_admm_update_l_cpu.jl")
include("cpu/mpacopf_admm_update_residual_cpu.jl")
include("cpu/mpacopf_admm_update_lz_cpu.jl")
include("cpu/mpacopf_admm_prepoststep_cpu.jl")

# GPU implementation for solving multi-period ACOPF
include("gpu/mpacopf_init_solution_gpu.jl")
include("gpu/mpacopf_eval_generator_kernel_gpu.jl")
include("gpu/mpacopf_tron_generator_kernel.jl")
include("gpu/mpacopf_auglag_generator_kernel_gpu.jl")
include("gpu/mpacopf_bus_kernel_gpu.jl")
include("gpu/mpacopf_admm_update_x_gpu.jl")
include("gpu/mpacopf_admm_update_xbar_gpu.jl")
include("gpu/mpacopf_admm_update_z_gpu.jl")
include("gpu/mpacopf_admm_update_l_gpu.jl")
include("gpu/mpacopf_admm_update_residual_gpu.jl")
include("gpu/mpacopf_admm_update_lz_gpu.jl")
include("gpu/mpacopf_admm_prepoststep_gpu.jl")

# PowerFlow solver on CPUs
include("cpu/pf_struct.jl")
include("cpu/pf_init_cpu.jl")
include("cpu/pf_eval_f_cpu.jl")
include("cpu/pf_eval_jac_cpu.jl")
include("cpu/pf_newton_raphson_cpu.jl")
include("cpu/pf_projection.jl")

# Rolling horizon
include("common/acopf_admm_rolling.jl")
include("cpu/acopf_admm_rolling_cpu.jl")
include("gpu/acopf_admm_rolling_gpu.jl")

# MPEC on CPUs
include("common/mpec_admm.jl")
include("common/mpec_admm_increment.jl")
include("cpu/mpec_init_solution_cpu.jl")
include("cpu/mpec_bus_kernel_cpu.jl")
include("cpu/mpec_admm_update_x_cpu.jl")
include("cpu/mpec_admm_update_xbar_cpu.jl")
include("cpu/mpec_admm_update_z_cpu.jl")
include("cpu/mpec_admm_update_l_cpu.jl")
include("cpu/mpec_admm_update_residual_cpu.jl")
include("cpu/mpec_admm_update_lz_cpu.jl")
include("cpu/mpec_admm_prepoststep_cpu.jl")

# MPEC on GPUs
include("gpu/mpec_init_solution_gpu.jl")
include("gpu/mpec_bus_kernel_gpu.jl")
include("gpu/mpec_admm_update_x_gpu.jl")
include("gpu/mpec_admm_update_xbar_gpu.jl")
include("gpu/mpec_admm_update_z_gpu.jl")
include("gpu/mpec_admm_update_l_gpu.jl")
include("gpu/mpec_admm_update_residual_gpu.jl")
include("gpu/mpec_admm_update_lz_gpu.jl")
include("gpu/mpec_admm_prepoststep_gpu.jl")

end # module
