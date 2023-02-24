"""
    admm_update_lz()
    
- update lz
- record time info.time_lz_update
- only used in two-level ADMM
- GPU kernel: update_lz_kernel
"""

function update_lz_kernel(n::Int, max_limit::Float64, z::CuDeviceArray{Float64,1}, lz::CuDeviceArray{Float64,1}, beta::Float64)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        lz[tx] = max(-max_limit, min(max_limit, lz[tx] + beta*z[tx]))
    end

    return
end

function admm_update_lz(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::ModelQpsub{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    par, sol, info = env.params, mod.solution, mod.info
    lztime = CUDA.@timed @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, par.beta)
    info.time_lz_update += lztime.time
    return
end