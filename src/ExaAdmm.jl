module ExaAdmm

using Printf
using FileIO
using DelimitedFiles
using LinearAlgebra
using SparseArrays
using MPI
using CUDA
import AMDGPU: ROCArray, has_rocm_gpu
using KernelAbstractions
using ExaTron
using Random

const KA = KernelAbstractions
## added by bowen
using Test
using JuMP
using Ipopt

export solve_acopf

struct KAArray{T} end

include("utils/parse_matpower.jl")
include("utils/opfdata.jl")
include("utils/environment.jl")
include("utils/grid_data.jl")
include("utils/print_statistics.jl")
include("utils/utilities_gpu.jl")
include("utils/utilities_ka.jl")

include("algorithms/admm_two_level.jl")
include("algorithms/admm_one_level.jl")

# ----------------------------------------
# A single period ACOPF implementation
# ----------------------------------------

# Interface to solve a single period ACOPF.
include("interface/solve_acopf.jl")
include("interface/solve_acopf_rolling.jl")
# Define "struct ModelAcopf" for encapsulating an ACOPF model.
include("models/acopf/acopf_model.jl")
include("models/acopf/acopf_admm_increment.jl")

# CPU specific implementation
include("models/acopf/acopf_init_solution_cpu.jl")
include("models/acopf/acopf_generator_kernel_cpu.jl")
include("models/acopf/acopf_eval_linelimit_kernel_cpu.jl")
include("models/acopf/acopf_auglag_linelimit_kernel_cpu.jl")
include("models/acopf/acopf_bus_kernel_cpu.jl")
include("models/acopf/acopf_admm_update_x_cpu.jl")
include("models/acopf/acopf_admm_update_xbar_cpu.jl")
include("models/acopf/acopf_admm_update_z_cpu.jl")
include("models/acopf/acopf_admm_update_l_cpu.jl")
include("models/acopf/acopf_admm_update_residual_cpu.jl")
include("models/acopf/acopf_admm_update_lz_cpu.jl")
include("models/acopf/acopf_admm_prepoststep_cpu.jl")

# CUDA specific implementation
include("models/acopf/acopf_init_solution_gpu.jl")
include("models/acopf/acopf_generator_kernel_gpu.jl")
include("models/acopf/acopf_eval_linelimit_kernel_gpu.jl")
include("models/acopf/acopf_tron_linelimit_kernel.jl")
include("models/acopf/acopf_auglag_linelimit_kernel_gpu.jl")
include("models/acopf/acopf_bus_kernel_gpu.jl")
include("models/acopf/acopf_admm_update_x_gpu.jl")
include("models/acopf/acopf_admm_update_xbar_gpu.jl")
include("models/acopf/acopf_admm_update_z_gpu.jl")
include("models/acopf/acopf_admm_update_l_gpu.jl")
include("models/acopf/acopf_admm_update_residual_gpu.jl")
include("models/acopf/acopf_admm_update_lz_gpu.jl")
include("models/acopf/acopf_admm_prepoststep_gpu.jl")

# KA specific implementation
include("models/acopf/acopf_init_solution_ka.jl")
include("models/acopf/acopf_generator_kernel_ka.jl")
include("models/acopf/acopf_eval_linelimit_kernel_ka.jl")
include("models/acopf/acopf_tron_linelimit_kernel_ka.jl")
include("models/acopf/acopf_auglag_linelimit_kernel_ka.jl")
include("models/acopf/acopf_bus_kernel_ka.jl")
include("models/acopf/acopf_admm_update_x_ka.jl")
include("models/acopf/acopf_admm_update_xbar_ka.jl")
include("models/acopf/acopf_admm_update_z_ka.jl")
include("models/acopf/acopf_admm_update_l_ka.jl")
include("models/acopf/acopf_admm_update_residual_ka.jl")
include("models/acopf/acopf_admm_update_lz_ka.jl")
include("models/acopf/acopf_admm_prepoststep_ka.jl")

# Rolling horizon
include("models/acopf/acopf_admm_rolling_cpu.jl")
include("models/acopf/acopf_admm_rolling_gpu.jl")
include("models/acopf/acopf_admm_rolling_ka.jl")

# ----------------------------------------
# Multi-period ACOPF implementation
# ----------------------------------------

# Interface to solve a multi-period ACOPF.
include("interface/solve_mpacopf.jl")
# Define "struct ModelMpacopf" for encapsulating an ACOPF model.
include("models/mpacopf/mpacopf_model.jl")
include("models/mpacopf/mpacopf_admm_increment.jl")

# CPU specific implementation
include("models/mpacopf/mpacopf_init_solution_cpu.jl")
include("models/mpacopf/mpacopf_eval_generator_kernel_cpu.jl")
include("models/mpacopf/mpacopf_auglag_generator_kernel_cpu.jl")
include("models/mpacopf/mpacopf_bus_kernel_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_x_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_xbar_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_z_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_l_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_residual_cpu.jl")
include("models/mpacopf/mpacopf_admm_update_lz_cpu.jl")
include("models/mpacopf/mpacopf_admm_prepoststep_cpu.jl")

# GPU specific implementation
include("models/mpacopf/mpacopf_init_solution_gpu.jl")
include("models/mpacopf/mpacopf_eval_generator_kernel_gpu.jl")
include("models/mpacopf/mpacopf_tron_generator_kernel.jl")
include("models/mpacopf/mpacopf_auglag_generator_kernel_gpu.jl")
include("models/mpacopf/mpacopf_bus_kernel_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_x_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_xbar_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_z_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_l_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_residual_gpu.jl")
include("models/mpacopf/mpacopf_admm_update_lz_gpu.jl")
include("models/mpacopf/mpacopf_admm_prepoststep_gpu.jl")

#=
# PowerFlow solver on CPUs
include("interface/solve_pf.jl")
include("models/pf/pf_struct.jl")
include("models/pf/pf_init_cpu.jl")
include("models/pf/pf_eval_f_cpu.jl")
include("models/pf/pf_eval_jac_cpu.jl")
include("models/pf/pf_projection.jl")

# MPEC on CPUs
include("interface/solve_mpec.jl")
include("models/mpec/mpec_admm_increment.jl")
include("models/mpec/mpec_init_solution_cpu.jl")
include("models/mpec/mpec_bus_kernel_cpu.jl")
include("models/mpec/mpec_admm_update_x_cpu.jl")
include("models/mpec/mpec_admm_update_xbar_cpu.jl")
include("models/mpec/mpec_admm_update_z_cpu.jl")
include("models/mpec/mpec_admm_update_l_cpu.jl")
include("models/mpec/mpec_admm_update_residual_cpu.jl")
include("models/mpec/mpec_admm_update_lz_cpu.jl")
include("models/mpec/mpec_admm_prepoststep_cpu.jl")

# MPEC on GPUs
include("models/mpec/mpec_init_solution_gpu.jl")
include("models/mpec/mpec_bus_kernel_gpu.jl")
include("models/mpec/mpec_admm_update_x_gpu.jl")
include("models/mpec/mpec_admm_update_xbar_gpu.jl")
include("models/mpec/mpec_admm_update_z_gpu.jl")
include("models/mpec/mpec_admm_update_l_gpu.jl")
include("models/mpec/mpec_admm_update_residual_gpu.jl")
include("models/mpec/mpec_admm_update_lz_gpu.jl")
include("models/mpec/mpec_admm_prepoststep_gpu.jl")
=#

# Interface to use ADMM solving QPsub.
include("interface/solve_qpsub.jl")

# Define "struct ModelAcopf" for encapsulating an ACOPF model.
include("models/qpsub/qpsub_model.jl")
include("models/qpsub/qpsub_admm_increment.jl")

# CPU
include("models/qpsub/qpsub_init_solution_cpu.jl")
include("models/qpsub/qpsub_generator_kernel_cpu.jl")
include("models/qpsub/qpsub_eval_linelimit_kernel_cpu.jl")
include("models/qpsub/qpsub_auglag_linelimit_kernel_cpu.jl")
include("models/qpsub/qpsub_bus_kernel_cpu.jl")
include("models/qpsub/qpsub_admm_update_x_cpu.jl")
include("models/qpsub/qpsub_admm_update_xbar_cpu.jl")
include("models/qpsub/qpsub_admm_update_z_cpu.jl")
include("models/qpsub/qpsub_admm_update_l_cpu.jl")
include("models/qpsub/qpsub_admm_update_l_cpu_single.jl")
include("models/qpsub/qpsub_admm_update_residual_cpu.jl")
include("models/qpsub/qpsub_admm_update_lz_cpu.jl")
include("models/qpsub/qpsub_admm_prepoststep_cpu.jl")

#CPU new
include("models/qpsub/qpsub_eval_Ab_linelimit_kernel_cpu.jl") #eval A,b for Ipopt and Tron
include("models/qpsub/qpsub_auglag_Ab_linelimit_kernel_cpu.jl") #all line
include("models/qpsub/qpsub_auglag_Ab_linelimit_kernel_cpu_ij.jl")  #line by line
include("models/qpsub/qpsub_auglag_tron_linelimit_kernel_cpu.jl") #Exatron build

# Others for individual use 




end # module
