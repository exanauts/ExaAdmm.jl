function set_rateA_kernel(nline::Int, param::CuDeviceArray{Float64,2}, rateA::CuDeviceArray{Float64,1})
  tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))

  if tx <= nline
      param[29,tx] = (rateA[tx] == 0.0) ? 1e3 : rateA[tx]
  end

  return
end

function acopf_set_linelimit(
  env::AdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
  mod::AbstractOPFModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
  info::IterationInformation
)
  CUDA.@sync @cuda threads=64 blocks=(div(mod.nline-1, 64)+1) set_rateA_kernel(mod.nline, mod.membuf, mod.rateA)
end
