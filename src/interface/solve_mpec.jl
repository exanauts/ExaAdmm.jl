function solve_acopf_mpec(case::String;
    case_format="matpower",
    outer_iterlim=20, inner_iterlim=1000, rho_pq=400.0, rho_va=40000.0,
    obj_scale=1.0, scale=1e-4, storage_ratio=0.0, storage_charge_max=1.0,
    use_gpu=false, use_linelimit=true, use_projection=false, tight_factor=0.99,
    outer_eps=2*1e-5, gpu_no=0, verbose=1
)
    T = Float64; TD = Array{Float64,1}; TI = Array{Int,1}; TM = Array{Float64,2}
    if use_gpu
        CUDA.device!(gpu_no)
        TD = CuArray{Float64,1}; TI = CuArray{Int,1}; TM = CuArray{Float64,2}
    end

    env = AdmmEnv{T,TD,TI,TM}(case, rho_pq, rho_va; case_format=case_format,
            use_gpu=use_gpu, use_linelimit=use_linelimit,
            use_projection=use_projection, tight_factor=tight_factor,
            storage_ratio=storage_ratio, storage_charge_max=storage_charge_max,
            gpu_no=gpu_no, verbose=verbose)
    mod = ComplementarityModel{T,TD,TI,TM}(env)

    env.params.scale = scale
    env.params.obj_scale = obj_scale
    env.params.outer_eps = outer_eps
    env.params.outer_iterlim = outer_iterlim
    env.params.inner_iterlim = inner_iterlim

    admm_two_level(env, mod)


    vm_dev = 0.0
    for i=1:mod.grid_data.ngen
        pg_idx = mod.gen_start + 2*(i-1)
        qg_idx = mod.gen_start + 2*(i-1) + 1
        vg_idx = mod.gen_start + 2*mod.grid.ngen + (i-1)
        fg_idx = mod.gen_start + 3*mod.grid.ngen + (i-1)
        vm_dev = max(vm_dev, abs(sqrt(mod.solution.u_curr[vg_idx]) - mod.grid.vm_setpoint[i]))
        #=
        @printf("[%3d]\n", i)
        @printf("   % 12.6e <= Pg = % 12.6e <= % 12.6e  Pg_setpoint + alpha_g x Fg = % 12.6e  Fg = % 12.6e\n",
                mod.grid.pgmin[i], mod.solution.u_curr[pg_idx], mod.grid.pgmax[i],
                mod.grid.pg_setpoint[i]+mod.grid.alpha[i]*mod.solution.u_curr[fg_idx], mod.solution.u_curr[fg_idx])
        @printf("   % 12.6e <= Qg = % 12.6e <= % 12.6e  Vg = % 12.6e Vg_setpoint = % 12.6e\n",
            mod.grid.qgmin[i], mod.solution.u_curr[qg_idx], mod.grid.qgmax[i],
            sqrt(mod.solution.u_curr[vg_idx]), (mod.grid.vgmin[i]+mod.grid.vgmax[i])/2)
        =#
    end

    @printf("Frequency change = % 12.6e\n", mod.solution.v_curr[mod.gen_start+3*mod.grid.ngen])
    @printf("|VM-VM^sp|_infty = % 12.6e\n", vm_dev)
    return env, mod
end
