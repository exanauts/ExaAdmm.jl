function admm_increment_outer(
    env::AdmmEnv,
    mod::UCMPModel,
    device=nothing
)
    mod.info.outer += 1
    admm_increment_outer(env, mod.mpmodel, device)
    return
end

function admm_increment_reset_inner(
    env::AdmmEnv,
    mod::UCMPModel,
    device=nothing
)
    mod.info.inner = 0
    admm_increment_reset_inner(env, mod.mpmodel, device)
    return
end

function admm_increment_inner(
    env::AdmmEnv,
    mod::UCMPModel,
    device=nothing
)
    mod.info.inner += 1
    mod.info.cumul += 1
    admm_increment_inner(env, mod.mpmodel, device)
    return
end