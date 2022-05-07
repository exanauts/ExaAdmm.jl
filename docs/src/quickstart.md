# Quick Start

Currently, `ExaAdmm.jl` supports electrical grid files in the MATLAB format. You can download them from [here](https://github.com/MATPOWER/matpower).
Below shows an example of solving `case1354pegase.m` using `ExaAdmm.jl` on GPUs.

```julia
using ExaAdmm
env, mod = ExaAdmm.solve_acopf("case1354pegase.m"; rho_pq=1e1, rho_va=1e3, outer_iterlim=20, inner_iterlim=20, scale=1e-4, tight_factor=0.99, use_gpu=true);
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
