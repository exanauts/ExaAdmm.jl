# Development Guide

This describes the steps to implement new model that can be solved by the algorithms provided in this package.

## Algorithm overview

Two-level ADMM algorithm can be implemented by defining the functions called in the while loop of `admm_two_level()` function in `algorithms/admm_two_level.jl` file.
These functions take two arguments of type `AdmmEnv` and `AbstractOPFModel`. You need to define your model structure that inherits `AbstractOPFModel` so that those functions can be called using multiple dispatching.

### Mathematical description

It is important to understand the algorithm implemented in this package before implementing one for your own model.

The algorithm aims at solving the following optimization problem

```math
\begin{align*}
\min_{x_i \in X_i,\bar{x} \in \bar{X},z} \quad & \sum_{i=1}^N f_i(x_i) \\
\text{subject to} \quad & x - \bar{x} + z = 0 & (\lambda) \\
& z = 0 & (\lambda_z) \\
& x:=(x_1,\dots,x_N)
\end{align*}
```

by using the form of augmented Lagrangian method as follows:

1. Initialize $\lambda_z$ and $\beta$.
2. Solve the following problem for given $(\lambda_z,\beta)$:

```math
\begin{align*}
\min_{x_i \in X_i,\bar{x} \in \bar{X},z} \quad & \sum_{i=1}^N f_i(x_i) + \lambda_z^T z + \frac{\beta}{2} \|z\|_2^2 \tag{AugLag} \\
\text{subject to} \quad & x - \bar{x} + z = 0 & (\lambda)
\end{align*}
```

3. Update $\lambda_z$
4. Repeat steps 2-3 until $z \approx 0$.

## Implementation steps

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
3. Implement all necessary functions: [A list of functions required to implement](@ref)

!!! note
    `ExaAdmm.jl` provides a default implementation for each function,
    dispatching on `AbstractOPFModel`. This default implementation
    matches the behavior of `ModelAcopf`, and allow the user to avoid
    overloading if the behavior of the new model `ModelMyopf`
    matches those of `ModelAcopf` for a particular function.


## A list of functions required to implement

Define the following functions that will take your new model structure:

### Functions in `myopf_admm_increment_outer.jl`

```@docs
ExaAdmm.admm_increment_outer(
    env::ExaAdmm.AdmmEnv,
    mod::ExaAdmm.AbstractOPFModel
)
ExaAdmm.admm_increment_reset_inner(
    env::ExaAdmm.AdmmEnv,
    mod::ExaAdmm.AbstractOPFModel
)
ExaAdmm.admm_increment_inner(
    env::ExaAdmm.AdmmEnv,
    mod::ExaAdmm.AbstractOPFModel
)
```

### Functions in `myopf_admm_prepoststep_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_outer_prestep(
   env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
   mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
ExaAdmm.admm_inner_prestep(
   env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
   mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
ExaAdmm.admm_poststep(
   env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
   mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_x_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_x(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_xbar_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_xbar(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_z_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_z(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_l_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_l(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_lz_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_lz(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

### Functions in `myopf_admm_update_residual_{cpu,gpu}.jl`

```@docs
ExaAdmm.admm_update_residual(
    env::ExaAdmm.AdmmEnv{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}},
    mod::ExaAdmm.AbstractOPFModel{Float64,Array{Float64,1},Array{Int,1},Array{Float64,2}}
)
```

!!! note
    Internally, these `x` and `xbar` correspond to `u` and `v` variables in the code, respectively.
    Variables `u` and `v` have the same structure. For example in ACOPF, the variable stores modeling variable values in the following order:
    ```math
    ({\color{blue} (p_{g,i}, q_{g,i})_{i=1,\dots,|G|}}, {\color{red} (p_{ij,\ell}, q_{ij,\ell}, p_{ji,\ell}, q_{ji,\ell}, w_{i,\ell}, w_{j,\ell}, t_{i,\ell}, t_{j,\ell})_{\ell=1,\dots,|L|}})
    ```

## Notes for GPU

To make your model run on Nvidia GPUs, the `AdmmEnv` and you also need to implement these functions that take arrays of type `CuArray`. You can find examples in `interface/solve_acopf.jl` and `interface/solve_mpacopf.jl`.
