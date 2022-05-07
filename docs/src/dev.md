# How to implement new model

This describes the steps to implement new model that can be solved by the algorithms provided in this package.

## Two-level ADMM algorithm

Two-level ADMM algorithm can be implemented by defining the functioins called in the while loop of `admm_two_level()` function in `algorithms/admm_two_level.jl` file.
These functions take two arguments of type `AdmmEnv` and `AbstractOPFModel`. You need to define your model structure that inherits `AbstractOPFModel` so that those functions can be called using multiple dispatching.

### Mathematical description

It is important to understand the algorithm implemented in this package before implementing one for your own model.

The algorithm aims at solving the following optimization problem

```math
\begin{align*}
\min_{x \in X,\bar{x} \in \bar{X},z} \quad & f(x) + g(\bar{x}) \\
\text{subject to} \quad & x - \bar{x} + z = 0 & (\lambda) \\
& z = 0 & (\lambda_z)
\end{align*}
```

by using the form of augmented Lagrangian method as follows:

1. Initialize $\lambda_z$ and $\beta$.
2. Solve the following problem for given $(\lambda_z,\beta)$:

```math
\begin{align*}
\min_{x \in X,\bar{x} \in \bar{X},z} \quad & f(x) + g(\bar{x}) + \lambda_z^T z + \frac{\beta}{2} \|z\|_2^2 \tag{AugLag} \\
\text{subject to} \quad & x - \bar{x} + z = 0 & (\lambda)
\end{align*}
```

3. Update $\lambda_z$
4. Repeat steps 2-3 until $z \approx 0$.

### Implementation steps

We encourage to follow our naming convention for the files created for your own model. In this description, we assume that we are creating our own model called `myopf`.
The steps required to implement the two-level ADMM algorithm for the model `myopf` are as follows:

1. Create a directory `models/myopf` for your own model.
2. Create a file `models/myopf/myopf_model.jl` to define the model structure from `AbstractOPFModel`.

   ```julia
   mutable struct ModelMyopf{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
       # ...
   end
   ```

   Examples are available in `models/acopf/acopf_model.jl` and `models/mpacopf_model.jl`.
3. Define the following functions that will take your new model structure:
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

- what is x, xbar, l, lz, etc.? Give the generic formulation and description.
  - x: variables for generators and lines.
  - xbar: variables for buses.
  - z: artificial variable that is driven to zero
  - l: multipliers for consensus constraints, `x - xbar + z= 0`
  - lz: multipliers for augmented Lagrangian for `z=0` constraint.
  - Internally, these `x` and `xbar` correspond to `u` and `v` variables in the code, respectively.
  - The structure of `u` and `v` variables is the same. They look like as follows:
    - |(pg_1,qg_1) | ... | (pg_|G|,qg_|G|)| (pij_1,qij_1,pji_1,qji_1,wi_1,wj_1,ti_1,tj_1) | ... | (pij_|L|,qij_|L|,pji_|L|,qji_|L|,wi_|L|,wj_|L|,ti_|L|,tj_|L|)|
    - In ACOPF, each generator has two variables (pg,qg) and each line has 8 variables. Therefore, the size of u and v is 2*|G| + 8*|L|. They are stored in the order of generators and lines, respectively.

