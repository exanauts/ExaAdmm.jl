function next_key(lines, pos)
    key = ""
    while pos < length(lines)
        terms = split(strip(lines[pos]))
        if !isempty(terms) &&
            terms[1] in ["mpc.baseMVA", "mpc.areas", "mpc.bus",
                         "mpc.gen", "mpc.branch", "mpc.gencost"]
            key = terms[1]
            break
        else
            pos += 1
        end
    end

    return key, pos
end

function get_field_names(key; case_format="matpower")
    @assert case_format in ["matpower", "pglib"]
    names = []

    if key == "mpc.bus"
        names = ["bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area",
                "Vm", "Va", "baseKV", "zone", "Vmax", "Vmin"]
    elseif key == "mpc.gen"
        if case_format == "matpower"
            names = ["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase",
                    "status", "Pmax", "Pmin", "Pc1", "Pc2", "Qc1min",
                    "Qc1max", "Qc2min", "Qc2max", "ramp_agc", "ramp_10",
                    "ramp_30", "ramp_q", "apf"]
        else
            names = ["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase",
                    "status", "Pmax", "Pmin" ]
        end
    elseif key == "mpc.branch"
        names = ["fbus", "tbus", "r", "x", "b", "rateA", "rateB",
                "rateC", "ratio", "angle", "status", "angmin", "angmax"]
    elseif key == "mpc.gencost"
        # Support quadratic cost only for now.
        names = ["cost_type", "startup", "shutdown", "n", "c2", "c1", "c0"]
    end

    return names
end

function read_raw_data(filename; case_format="matpower")
    f = open(filename, "r")
    lines = readlines(f)
    p = 1
    data = Dict()
    while p <= length(lines)
        key, p = next_key(lines, p)
        p += 1

        if key == ""
            break
        elseif key == "mpc.baseMVA"
            data["mpc.baseMVA"] = parse(Float64,replace(strip(split(lines[p-1], "=")[2]), ";"=>""))
        else
            q = p
            while !startswith(strip(lines[q]), "];")
                q += 1
            end
            data[key] = [Dict{String,Any}() for j=p:q-1]
            names = get_field_names(key; case_format=case_format)
            for j=1:q-p
                comment = findfirst('%', lines[p+j-1])
                line_of_interest = (comment === nothing) ? strip(lines[p+j-1]) : strip(lines[p+j-1][1:comment-1])
                elements = split(line_of_interest)
                for (k,v) in enumerate(names)
                    data[key][j][v] = parse(Float64, replace(strip(elements[k]), ";"=>""))
                end
                if comment !== nothing
                    data[key][j]["type"] = strip(lines[p+j-1][comment+1:end])
                end
            end
        end
    end
    close(f)

    data["baseMVA"] = data["mpc.baseMVA"]
    data["case_format"] = case_format
    return data
end

function add_bus(data)
    nbus = length(data["mpc.bus"])
    data["bus"] = [Dict{String,Float64}() for i=1:nbus]
    data["bus2idx"] = Dict{Int,Int}()
    data["idx2bus"] = Dict{Int,Int}()
    busref = Int[]
    for i=1:nbus
        data["bus2idx"][Int(data["mpc.bus"][i]["bus_i"])] = i
        data["idx2bus"][i] = Int(data["mpc.bus"][i]["bus_i"])
        if Int(data["mpc.bus"][i]["type"]) == 3
            push!(busref, Int(data["mpc.bus"][i]["bus_i"]))
        end
    end

    if isempty(busref)
        println("Error: reference bus was not found.")
        exit(-1)
    end
    data["busref"] = busref

    bf_name = get_field_names("mpc.bus"; case_format=data["case_format"])
    for i=1:nbus
        for v in bf_name
            data["bus"][i][v] = data["mpc.bus"][i][v]
            if v == "Va"
                data["bus"][i][v] *= (pi/180.0)
            end
        end
    end
end

function add_gen(data)
    ngen = length(data["mpc.gen"])
    data["gen"] = Vector{Dict{String,Float64}}()
    data["gencost"] = Vector{Dict{String,Float64}}()
    data["gen2idx"] = Dict{Int,Int}(g => -1 for g=1:ngen)
    data["idx2gen"] = Dict{Int,Int}()
    data["busgen"] = [Int[] for i=1:length(data["bus"])]

    gf_name = get_field_names("mpc.gen"; case_format=data["case_format"])
    gc_name = get_field_names("mpc.gencost"; case_format=data["case_format"])
    j = 1
    for g=1:ngen
        if Int(data["mpc.gen"][g]["status"]) != 1
            continue
        end
        data["gen2idx"][g] = j
        data["idx2gen"][j] = g
        push!(data["busgen"][data["bus2idx"][Int(data["mpc.gen"][g]["bus"])]], j)
        push!(data["gen"], Dict{String,Float64}())
        push!(data["gencost"], Dict{String,Float64}())
        for v in gf_name
            data["gen"][j][v] = data["mpc.gen"][g][v]
            if v in ["Pg", "Qg", "Qmax", "Qmin", "Pmax", "Pmin"]
                data["gen"][j][v] /= data["baseMVA"]
            end
        end
        for v in gc_name
            data["gencost"][j][v] = data["mpc.gencost"][g][v]
        end
        j += 1
    end
end

function add_branch(data)
    nbr = length(data["mpc.branch"])
    data["branch"] = Vector{Dict{String,Float64}}()
    lf_name = get_field_names("mpc.branch"; case_format=data["case_format"])
    j = 1
    for i=1:nbr
        if Int(data["mpc.branch"][i]["status"]) != 1
            continue
        end
        push!(data["branch"], Dict{String,Float64}())
        for v in lf_name
            data["branch"][j][v] = data["mpc.branch"][i][v]
            #=
            if v in ["rateA", "rateB", "rateC"]
                data["branch"][j][v] /= data["baseMVA"]
            end
            =#
        end
        j += 1
    end
end

function add_admittance_shunt(data)
    data["YffR"] = Float64[]; data["YffI"] = Float64[]
    data["YttR"] = Float64[]; data["YttI"] = Float64[]
    data["YftR"] = Float64[]; data["YftI"] = Float64[]
    data["YtfR"] = Float64[]; data["YtfI"] = Float64[]

    for k=1:length(data["branch"])
        fr = data["bus2idx"][Int(data["branch"][k]["fbus"])]
        to = data["bus2idx"][Int(data["branch"][k]["tbus"])]
        r = data["branch"][k]["r"]
        x = data["branch"][k]["x"]
        b = data["branch"][k]["b"]
        tap = data["branch"][k]["ratio"]
        angle = data["branch"][k]["angle"]

        if tap == 0.0
            tap = 1.0
        end
        tap *= exp((angle*(pi/180.0))*1im)

        Ys = 1 / (r + x*1im)
        Ytt = Ys + (0.5*b)*1im
        Yff = Ytt / (conj(tap)*tap)
        Yft = -Ys / conj(tap)
        Ytf = -Ys / tap

        push!(data["YffR"], real(Yff)); push!(data["YffI"], imag(Yff))
        push!(data["YttR"], real(Ytt)); push!(data["YttI"], imag(Ytt))
        push!(data["YftR"], real(Yft)); push!(data["YftI"], imag(Yft))
        push!(data["YtfR"], real(Ytf)); push!(data["YtfI"], imag(Ytf))
    end
end

function add_mapping_branch_to_bus(data)
    data["frombus"] = [Int[] for i=1:length(data["bus"])]
    data["tobus"] = [Int[] for i=1:length(data["bus"])]
    for i=1:length(data["branch"])
        fr = data["bus2idx"][Int(data["branch"][i]["fbus"])]
        to = data["bus2idx"][Int(data["branch"][i]["tbus"])]
        push!(data["frombus"][fr], i)
        push!(data["tobus"][to], i)
    end
end

function parse_matpower(filename; case_format="matpower")
    data = read_raw_data(filename; case_format=case_format)
    case = split(basename(filename),".")[1]

    add_bus(data)
    add_gen(data)
    add_branch(data)
    add_admittance_shunt(data)
    add_mapping_branch_to_bus(data)

    br_stat = [length(x)+length(y) for (x,y) in zip(data["frombus"],data["tobus"])]
    thresholds = 10

    @printf(" ** Statistics of %s\n", case)
    @printf("  # buses     : %5d\n", length(data["bus"]))
    @printf("  # generators: %5d (%5d active)\n", length(data["mpc.gen"]), length(data["gen"]))
    @printf("  # branches  : %5d (%5d active)\n", length(data["mpc.branch"]), length(data["branch"]))
    @printf("  # gencost   : %5d (%5d active)\n", length(data["mpc.gen"]), length(data["gen"]))
    #=
    @printf("  # max branches      per bus: %5d\n", maximum(br_stat))
    @printf("  # min branches      per bus: %5d\n", minimum(br_stat))
    @printf("  # avg branches      per bus: %5.2f\n", mean(br_stat))
    @printf("  # std branches      per bus: %5.2f\n", std(br_stat))
    @printf("  # median branches   per bus: %5.2f\n", median(br_stat))
    @printf("  # buses with <= %2d branches: %5d (%.2f%%)\n",
            thresholds,
            length(findall(x->x<=thresholds, br_stat)),
            100*(length(findall(x->x<=thresholds, br_stat))/length(data["bus"])))
    =#

    return data
end