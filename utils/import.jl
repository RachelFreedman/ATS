module Import

using LinearAlgebra, Serialization, Statistics, ParticleFilters

export State

struct State
    u::Array{Float64}         # list of N utility values for N items
    d::Array{Array{Float64}}  # list of K arm distributions, each assigning probabilities to N items
    b::Array{Float64}         # list of M beta values
end

function parse_state(s::String)
    sp = split(s, ['{', '}'])

    u_str = split(sp[1], ['[', ']'])[2]
    u = [parse(Float64, x) for x in split(u_str, ", ")]

    spsp = split(sp[3], ['[', ']'])
    d_str = [spsp[3], spsp[5], spsp[7]]
    d = [[parse(Float64, x) for x in split(elem, ", ")] for elem in d_str]

    b_str = spsp[10]
    b = [parse(Float64, x) for x in split(b_str, ", ")]

    return State(u,d,b)
end

function get_optimal_arm(s::State)
    max_arm = -1
    max_val = -1
    for i in 1:length(s.d)
        val_i = dot(s.u, s.d[i])
        if val_i > max_val
            max_arm = i
            max_val = val_i
        end
    end
    return "C"*string(max_arm), max_val
end

function get_star(expID::String, runs::Int, directory)
    lines = []
    run_lines = []
    for run in 1:runs
        run_lines = []
        open(directory*"/sims/"*expID*"_run"*string(run)*".txt", "r") do file
            for line in readlines(file)
                push!(run_lines, line)
            end
        end
        push!(lines, run_lines)
    end
    
    sims = [[split(l,',') for l in run_lines[2:end]] for run_lines in lines]

    s = [parse_state(l[1]) for l in lines]
    t = [[parse(Int64, s[1]) for s in sim] for sim in sims]
    a = [[s[3] for s in sim] for sim in sims]
    r = [[parse(Float64, s[end]) for s in sim] for sim in sims]
    
    return s,t,a,r
end

function print_action_hist(a::Vector{Vector{SubString{String}}}, actions)
    first_action_hist = zeros(5)
    second_action_hist = zeros(5)
    third_action_hist = zeros(5)
    fourth_action_hist = zeros(5)

    for ai in a
        for i in 1:length(actions)
            if ai[1] == actions[i]
                first_action_hist[i] = first_action_hist[i] + 1
            end
            if ai[2] == actions[i]
                second_action_hist[i] = second_action_hist[i] + 1
            end
            if ai[3] == actions[i]
                third_action_hist[i] = third_action_hist[i] + 1
            end
            if ai[4] == actions[i]
                fourth_action_hist[i] = fourth_action_hist[i] + 1
            end
        end
    end

    println("First Action")
    for i in 1:length(actions)
        println(actions[i], ":\t", first_action_hist[i])
    end

    println("\nSecond Action")
    for i in 1:length(actions)
        println(actions[i], ":\t", second_action_hist[i])
    end

    println("\nThird Action")
    for i in 1:length(actions)
        println(actions[i], ":\t", third_action_hist[i])
    end
end

function get_avg_belief(beliefs::Matrix{Array{ParticleCollection{State}}})
    runs = size(beliefs)[1]
    final_states = [mode(beliefs[run][end]) for run in 1:runs]
    final_state_belief = Array{Array{Float64}}(undef, runs)
    for run in 1:runs
        wt_run = Array{Float64}(undef, length(beliefs[run]))
        for i in 1:length(beliefs[run])
            wt_run[i] = pdf(beliefs[run][i], final_states[run])
        end
        final_state_belief[run] = wt_run
    end
    avg_belief = [mean([x[i] for x in final_state_belief]) for i in 1:length(final_state_belief[1])]
    return avg_belief
end 

function get_avg_belief(beliefs)
    runs = size(beliefs)[1]
    final_states = [mode(beliefs[run][end]) for run in 1:runs]
    final_state_belief = Array{Array{Float64}}(undef, runs)
    for run in 1:runs
        wt_run = Array{Float64}(undef, length(beliefs[run]))
        for i in 1:length(beliefs[run])
            wt_run[i] = pdf(beliefs[run][i], final_states[run])
        end
        final_state_belief[run] = wt_run
    end
    avg_belief = [mean([x[i] for x in final_state_belief]) for i in 1:length(final_state_belief[1])]
    return avg_belief
end 

function get_avg_belief_marginalized_across_d(beliefs::Matrix{Array{ParticleCollection{State}}})
    runs = size(beliefs)[1]
    final_us = [mode(beliefs[run][end]).u for run in 1:runs]
    final_us_belief = Array{Array{Float64}}(undef, runs)
    for run in 1:runs
        u_probdicts = marginalize_across_d(beliefs[run])
        run_u = final_us[run]
        final_us_belief[run] = [get(prob, run_u, 0) for prob in u_probdicts]
    end
    avg_belief = [mean([x[i] for x in final_us_belief]) for i in 1:length(final_us_belief[1])]
    return avg_belief
end  

function get_final_states(beliefs::Matrix{Array{ParticleCollection{State}}})
    return [mode(b[end]) for b in beliefs]
end

function print_state(s::State)
    println("\t u: ", s.u)
    println("\t d1: ", s.d[1], "\t (exp val ", dot(s.u, true_state.d[1]), ")")
    println("\t d1: ", s.d[2], "\t (exp val ", dot(s.u, true_state.d[2]), ")")
    println("\t d1: ", s.d[3], "\t (exp val ", dot(s.u, true_state.d[3]), ")")
end

function marginalize_across_d(beliefs::Array{ParticleCollection{State}})
    u_probdicts = Array{Dict}(UndefInitializer(), length(beliefs))

    for i in 1:length(beliefs)
        b = beliefs[i]
        u_probdict = Dict()
        s_probdict = ParticleFilters.probdict(b)
        for s_key in keys(s_probdict)
            u_key = s_key.u
            if u_key in keys(u_probdict)
                u_probdict[u_key] = u_probdict[u_key] + s_probdict[s_key]
            else
                u_probdict[u_key] = s_probdict[s_key]
            end
        end
        u_probdicts[i] = u_probdict
    end
    return u_probdicts
end

function import_experiment(expID::String, runs, directory="..")
    s, t, a, r = get_star(expID, runs, directory)
    beliefs = deserialize(open(directory*"/beliefs/"*expID*"_belief.txt", "r"))
    final_states = [mode(beliefs[run][end]) for run in 1:size(beliefs)[1]]
    avg_belief = get_avg_belief(beliefs)
    return s, t, a, r, beliefs, final_states, avg_belief
end

function import_IDs(IDs::Vector{String}, runs::Int; directory="..")
    s = Vector{Any}()
    t = Vector{Vector{Any}}()
    a = Vector{Vector{Any}}()
    r = Vector{Vector{Float64}}()
    for id in IDs
        s_, t_, a_, r_ = get_star(id, runs, directory)
        push!(s, s_[1])
        append!(a, a_)
        append!(r, r_)
        append!(t, t_)
    end
    return s, t, a, r
end

end