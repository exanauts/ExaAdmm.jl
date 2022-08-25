function admm_increment_outer(
    env::AdmmEnv,
    mod::ModelMpacopf,
    device::Nothing=nothing
)
    mod.info.outer += 1
    for i=1:mod.len_horizon
        admm_increment_outer(env, mod.models[i])
    end
    return
end

function admm_increment_reset_inner(
    env::AdmmEnv,
    mod::ModelMpacopf,
    device::Nothing=nothing
)
    mod.info.inner = 0
    for i=1:mod.len_horizon
        admm_increment_reset_inner(env, mod.models[i])
    end
    return
end

function admm_increment_inner(
    env::AdmmEnv,
    mod::ModelMpacopf,
    device::Nothing=nothing
)
    mod.info.inner += 1
    mod.info.cumul += 1
    for i=1:mod.len_horizon
        admm_increment_inner(env, mod.models[i])
    end
    return
end