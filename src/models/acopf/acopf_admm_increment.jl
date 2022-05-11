"""
Increment outer iteration counter by one.
"""
function admm_increment_outer(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.outer += 1
    return
end

"""
Reset inner iteration counter to zero.
"""
function admm_increment_reset_inner(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.inner = 0
    return
end

"""
Increment inner iteration counter by one.
"""
function admm_increment_inner(
    env::AdmmEnv,
    mod::ModelAcopf
)
    mod.info.inner += 1
    mod.info.cumul += 1
    return
end