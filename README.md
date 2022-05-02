# ExaAdmm.jl
[![Run tests](https://github.com/exanauts/ExaAdmm.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/exanauts/ExaAdmm.jl/actions/workflows/ci.yml)

ExaAdmm.jl implements the two-level alternating direction method of multipliers for solving the component-based decomposition of alternating current optimal power flow problems on GPUs.

## How to install

We have tested `ExaAdmm.jl` using `Julia@v1.6.3` and `CUDA.jl@v3.4.2`.

```
$ git clone https://github.com/exanauts/ExaAdmm.jl
$ cd ExaAdmm
$ julia --project deps/deps.jl
```

`deps.jl` installs [ExaTron.jl](https://github.com/exanauts/ExaTron.jl/tree/youngdae/multiperiod), a GPU-based batch solver for nonlinear nonconvex problems.
Details of `ExaTron.jl` are described in [our technical report](https://arxiv.org/abs/2106.14995).

## How to run

Currently, `ExaAdmm.jl` supports electrical grid files in the MATLAB format. You can download them from [here](https://github.com/MATPOWER/matpower).
Below shows an example of solving `case1354pegase.m` using `ExaAdmm.jl` on GPUs.

```
$ julia --project
julia> using ExaAdmm
julia> env, mod = ExaAdmm.solve_acopf("case1354pegase.m"; rho_pq=1e1, rho_va=1e3, outer_iterlim=20, inner_iterlim=20, scale=1e-4, tight_factor=0.99, use_gpu=true);
```

The following table shows parameter values we used for solving pegase and ACTIVSg data.
Data | rho_pq | rho_va | scale | obj_scale
---- | ------ | ------ | ----- | ---------
1354pegase | 1e1 | 1e3 | 1e-4 | 1.0
2869pegase | 1e1 | 1e3 | 1e-4 | 1.0
9241pegase | 5e1 | 5e3 | 1e-4 | 1.0
13659pegase | 5e1 | 5e3 | 1e-4 | 1.0
ACTIVSg25k | 3e3 | 3e4 | 1e-5 | 1.0
ACTIVSg70k | 3e4 | 3e5 | 1e-5 | 2.0

We have used the same `tight_factor=0.99`, `outer_iterlim=20`, and `inner_iterlim=1000` for all of the above data.

## Acknowledgements

This research was supported by the Exascale ComputingProject (17-SC-20-SC),  a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration.
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
