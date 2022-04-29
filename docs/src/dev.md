# How to implement new model

This describes the steps to implement new model that can be solved by the algorithms provided in this package.

Basically, by defining functioins called in the while loop of `admm_two_level()` function in `algorithms/admm_two_level.jl` file, you can use the two-level ADMM algorithm. These functions take two arguments of type `AdmmEnv` and `AbstractOPFModel`, respectively. You need to define your model structure that inherits `AbstractOPFModel` so that those functions can be called using multiple dispatching. So the steps are
1. Define your model structure. Examples are `models/acopf_model.jl` and `models/mpacopf_model.jl`.
2. Define the following functions that will take your new model structure:
  - `admm_increrment_outer`
  - `admm_outer_prestep`
  - `admm_increment_reset_inner`
  - `admm_increment_inner`
  - `admm_inner_prestep`
  - `admm_update_x`
  - `admm_update_xbar`
  - `admm_update_z`
  - `admm_update_l`
  - `admm_update_residual`
  - `admm_update_lz`
  - `admm_poststep`

Please see `models` directory to check out existing implementation. Note that to make your model run on Nvidia GPUs, the `AdmmEnv` and you also need to implement these functions that take arrays of type `CuArray`. You can find examples in `interface/solve_acopf.jl` and `interface/solve_mpacopf.jl`.

- naming convention
- model struct
- files to create
- cpu vs gpu?
- what is x, xbar, l, lz, etc.? Give the generic formulation and description.
  - x: variables for generators and lines.
  - xbar: variables for buses.
  - z: artificial variable that is drived to zero
  - l: multipliers for consensus constraints, `x - xbar + z= 0`
  - lz: multipliers for augmented Lagrangian for `z=0` constraint.
  - Internally, these `x` and `xbar` correspond to `u` and `v` variables in the code, respectively.
  - The structure of `u` and `v` variables is the same. They look like as follows:
    - |(pg_1,qg_1) | ... | (pg_|G|,qg_|G|)| (pij_1,qij_1,pji_1,qji_1,wi_1,wj_1,ti_1,tj_1) | ... | (pij_|L|,qij_|L|,pji_|L|,qji_|L|,wi_|L|,wj_|L|,ti_|L|,tj_|L|)|
    - In ACOPF, each generator has two variables (pg,qg) and each line has 8 variables. Therefore, the size of u and v is 2*|G| + 8*|L|. They are stored in the order of generators and lines, respectively.

