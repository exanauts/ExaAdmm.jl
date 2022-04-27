function acopf_admm_increment_outer(
    env::UCAdmmEnv,
    mod::UCMPModel
)
    mod.info.outer += 1
    for i=1:mod.len_horizon
        mod.models[i].info.outer += 1
    end
    return
end

function acopf_admm_increment_reset_inner(
    env::UCAdmmEnv,
    mod::UCMPModel
)
    mod.info.inner = 0
    for i=1:mod.len_horizon
        mod.models[i].info.inner = 0
    end
    return
end

function acopf_admm_increment_inner(
    env::UCAdmmEnv,
    mod::UCMPModel
)
    mod.info.inner += 1
    mod.info.cumul += 1
    for i=1:mod.len_horizon
        mod.models[i].info.inner += 1
        mod.models[i].info.cumul += 1
    end
    return
end