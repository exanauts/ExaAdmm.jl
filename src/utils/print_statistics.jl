function print_statistics(
    env::AdmmEnv,
    mod::AbstractOPFModel
)
    info = mod.info

    @printf(" ** Statistics\n")
    @printf("Objective value  . . . . . . . . . %12.6e\n", info.objval)
    @printf("Residual . . . . . . . . . . . . . %12.6e\n", info.mismatch)
    @printf("Outer iterations . . . . . . . . . %12d\n", info.outer)
    @printf("Cumulative iterations  . . . . . . %12d\n", info.cumul)
    @printf("Time per iteration . . . . . . . . %12.3f (secs/iter)\n", info.time_overall / info.cumul)
    @printf("Overall time . . . . . . . . . . . %12.3f (secs)\n", info.time_overall + info.time_projection)
    @printf("Projection time  . . . . . . . . . %12.3f (secs)\n", info.time_projection)
    @printf("Generator time . . . . . . . . . . %12.3f (secs)\n", info.user.time_generators)
    @printf("Branch time. . . . . . . . . . . . %12.3f (secs)\n", info.user.time_branches)
    @printf("Bus time . . . . . . . . . . . . . %12.3f (secs)\n", info.user.time_buses)
    @printf("G+Br+B time. . . . . . . . . . . . %12.3f (secs)\n",
            info.user.time_generators+info.user.time_branches+info.user.time_buses)
    return
end