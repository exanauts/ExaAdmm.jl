using ExaAdmm

# Functions for collecting solutions
function collect_ucmpmodel_solution(ucmpmodel::ExaAdmm.UCMPModel)
    T = ucmpmodel.mpmodel.len_horizon
    ngen = ucmpmodel.mpmodel.models[1].grid_data.ngen
    nbus = ucmpmodel.mpmodel.models[1].grid_data.nbus
    nline = ucmpmodel.mpmodel.models[1].grid_data.nline
    pg = zeros(T, ngen)
    qg = zeros(T, ngen)
    for t in 1:T
        for i in 1:ngen
            pg[t,i] = ucmpmodel.mpmodel.models[t].solution.u_curr[2*i-1]
            qg[t,i] = ucmpmodel.mpmodel.models[t].solution.u_curr[2*i]
        end
    end
    return pg, qg
end

function collect_mpmodel_solution(mpmodel::ExaAdmm.ModelMpacopf)
    T = mpmodel.len_horizon
    ngen = mpmodel.models[1].grid_data.ngen
    nbus = mpmodel.models[1].grid_data.nbus
    nline = mpmodel.models[1].grid_data.nline
    mppg = zeros(T, ngen)
    mpqg = zeros(T, ngen)
    for t in 1:T
        for i in 1:ngen
            mppg[t,i] = mpmodel.models[t].solution.u_curr[2*i-1]
            mpqg[t,i] = mpmodel.models[t].solution.u_curr[2*i]
        end
    end
    return mppg, mpqg
end


# MPACOPF: case9
# case = "case9"
# file = "./data/$(case).m"
# load_file = "./data/$(case)_onehour_168"
# gen_prefix = "./data/$(case)_gen"
# env, mpmodel = ExaAdmm.solve_mpacopf(file, load_file; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, outer_iterlim=100, inner_iterlim=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=false, warm_start=false); # FOR CPU DEBUGGING
# env, mpmodel = ExaAdmm.solve_mpacopf(file, load_file; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, outer_iterlim=100, inner_iterlim=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

# MPACOPF: case9 v2
# case = "case9"
# file = "../../../UC-ACOPF/data_v2/$(case).m"
# load_file = "../../../UC-ACOPF/data_v2/multiperiod_data/$(case)_onehour_168"
# gen_prefix = "../../../UC-ACOPF/data_v2/multiperiod_data/$(case)_gen"
# env, mpmodel = ExaAdmm.solve_mpacopf(file, load_file; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, outer_iterlim=100, inner_iterlim=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=false, warm_start=false); # FOR CPU DEBUGGING
# env, mpmodel = ExaAdmm.solve_mpacopf(file, load_file; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, outer_iterlim=100, inner_iterlim=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

# case9
# case = "case9"
# file = "./data/$(case).m"
# load_file = "./data/$(case)_onehour_168"
# gen_prefix = "./data/$(case)_gen"
# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=false, warm_start=false); # FOR CPU DEBUGGING
# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

# case9 v2
case = "case9"
file = "../UC-ACOPF/data_v2/$(case).m"
load_file = "../UC-ACOPF/data_v2/multiperiod_data/$(case)_onehour_168"
gen_prefix = "../UC-ACOPF/data_v2/multiperiod_data/$(case)_gen"

# WARM-UP
# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=3, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=false, warm_start=false); # FOR CPU DEBUGGING
# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=1e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

# case118 v2
case = "case118"
file = "../UC-ACOPF/data_v2/$(case).m"
load_file = "../UC-ACOPF/data_v2/multiperiod_data/$(case)_onehour_168"
gen_prefix = "../UC-ACOPF/data_v2/multiperiod_data/$(case)_gen"
# env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=5e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=false, warm_start=false); # FOR CPU DEBUGGING

using CUDA
CUDA.device!(1)
env, ucmpmodel = ExaAdmm.solve_ucmp(file, load_file, gen_prefix; ramp_ratio=0.1, rho_pq=5e3, rho_va=1e4, rho_uc=1e4, outer_iterlim=100, inner_iterlim=50, max_auglag=50, start_period=1, end_period=24, scale=1e-4, tight_factor=0.99, use_gpu=true, warm_start=false); # FOR GPU DEBUGGING

println("DONE")
