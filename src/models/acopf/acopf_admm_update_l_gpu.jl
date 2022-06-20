function update_l_kernel(
    n::Int, l::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    lz::CuDeviceArray{Float64,1}, beta::Float64
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

    if tx <= n
        @inbounds begin
            l[tx] = -(lz[tx] + beta*z[tx])
        end
    end

    return
end

function admm_update_l(
    env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::AbstractOPFModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    device::Nothing=nothing
)
    par, sol, info = env.params, mod.solution, mod.info
    ltime = CUDA.@timed @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_l_kernel(mod.nvar, sol.l_curr, sol.z_curr, sol.lz, par.beta)
    info.time_l_update += ltime.time
    return
end
