abstract type AbstractGridData{T,TD,TI,TM} end

mutable struct GridData{T,TD,TI,TM} <: AbstractGridData{T,TD,TI,TM}
    baseMVA::T
    droop::T
    ngen::Int
    nline::Int
    nbus::Int
    nstorage::Int

    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    vgmin::TD
    vgmax::TD
    alpha::TD
    ramp_rate::TD
    pg_setpoint::TD
    vm_setpoint::TD
    c2::TD
    c1::TD
    c0::TD
    YshR::TD
    YshI::TD
    YffR::TD
    YffI::TD
    YftR::TD
    YftI::TD
    YttR::TD
    YttI::TD
    YtfR::TD
    YtfI::TD
    FrVmBound::TD
    ToVmBound::TD
    FrVaBound::TD
    ToVaBound::TD
    rateA::TD
    FrStart::TI
    FrIdx::TI
    ToStart::TI
    ToIdx::TI
    GenStart::TI
    GenIdx::TI
    Pd::TD
    Qd::TD
    Vmin::TD
    Vmax::TD
    brBusIdx::TI

    StoStart::TI
    StoIdx::TI
    chg_min::TD
    chg_max::TD
    eta_chg::TD
    eta_dis::TD
    energy_min::TD
    energy_max::TD
    energy_setpoint::TD

    function GridData{T,TD,TI,TM}(env::AdmmEnv{T,TD,TI,TM}) where {T,TD<:AbstractArray{T},TI<:AbstractArray{Int},TM<:AbstractArray{T,2}}
        grid = new{T,TD,TI,TM}()
        grid.baseMVA = env.data.baseMVA
        grid.pgmin, grid.pgmax, grid.qgmin, grid.qgmax, grid.c2, grid.c1, grid.c0 = get_generator_data(env.data; use_gpu=env.use_gpu)
        grid.YshR, grid.YshI, grid.YffR, grid.YffI, grid.YftR, grid.YftI,
            grid.YttR, grid.YttI, grid.YtfR, grid.YtfI,
            grid.FrVmBound, grid.ToVmBound,
            grid.FrVaBound, grid.ToVaBound, grid.rateA = get_branch_data(env.data; use_gpu=env.use_gpu, tight_factor=env.tight_factor)
        grid.FrStart, grid.FrIdx, grid.ToStart, grid.ToIdx, grid.GenStart, grid.GenIdx, grid.Pd, grid.Qd, grid.Vmin, grid.Vmax = get_bus_data(env.data; use_gpu=env.use_gpu)
        grid.brBusIdx = get_branch_bus_index(env.data; use_gpu=env.use_gpu)
        grid.vgmin, grid.vgmax, grid.vm_setpoint = get_generator_bus_data(env.data; use_gpu=env.use_gpu)
        grid.droop = env.droop
        grid.alpha, grid.pg_setpoint = get_generator_primary_control(env.data; droop=env.droop, use_gpu=env.use_gpu)
        grid.chg_min, grid.chg_max, grid.energy_min, grid.energy_max, grid.energy_setpoint, grid.eta_chg, grid.eta_dis = get_storage_data(env.data; use_gpu=env.use_gpu)
        grid.StoIdx, grid.StoStart = get_bus_storage_index(env.data; use_gpu=env.use_gpu)

        grid.ngen = length(env.data.generators)
        grid.nline = length(env.data.lines)
        grid.nbus = length(env.data.buses)
        grid.nstorage = length(env.data.storages)

        return grid
    end
end
