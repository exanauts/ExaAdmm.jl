# ExaAdmm

ExaAdmm.jl implements the two-level alternating direction method of multipliers for solving the component-based decomposition of alternating current optimal power flow problems on GPUs.

## How to install

We have tested `ExaAdmm.jl` using `Julia@v1.6.3` and `CUDA.jl@v3.4.2`.

```shell
git clone https://github.com/exanauts/ExaAdmm.jl
cd ExaAdmm
julia --project deps/deps.jl
```

`deps.jl` installs [ExaTron.jl](https://github.com/exanauts/ExaTron.jl/tree/youngdae/multiperiod), a GPU-based batch solver for nonlinear nonconvex problems.
Details of `ExaTron.jl` are described in [our technical report](https://arxiv.org/abs/2106.14995).

## Acknowledgements

This research was supported by the Exascale ComputingProject (17-SC-20-SC),  a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
