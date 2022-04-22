function admm_increment_outer(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.outer += 1
    return
end

function admm_increment_reset_inner(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.inner = 0
    return
end

function admm_increment_inner(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.inner += 1
    mod.info.cumul += 1
    return
end