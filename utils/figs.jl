module Figs

using Plots, Statistics, NaNStatistics, LinearAlgebra

function plot_avg_r_multiple_experiments(r::Vector{Vector{Vector{Float64}}}, granularity::Int, y_list, title)
    avg_r = calc_avg_r_multiple_experiments(r, granularity)
    avg_r_array = [avg_r[i,1,:] for i in 1:size(avg_r)[1]]
    
    x = collect(range(granularity, length(r[1][1]), floor(Int, length(r[1][1])/granularity)))
    
    plot(x, avg_r_array, 
        ylims = (0,10),
        xlabel = "timestep",
        ylabel = "reward",
        title = title, 
        labels=y_list)
end

function plot_avg_r_multiple_experiments_normalized(r::Vector{Vector{Any}}, granularity::Int, y_list, title)
    avg_r = calc_avg_r_multiple_experiments(r, granularity)
    avg_r_array = [avg_r[i,1,:] for i in 1:size(avg_r)[1]]
    
    x = collect(range(granularity, length(r[1][1]), floor(Int, length(r[1][1])/granularity)))
    
    plot(x, avg_r_array, 
        ylims = (0,1.),
        xlabel = "timestep",
        ylabel = "reward",
        title = title, 
        labels=y_list)
end

function plot_avg_r_multiple_experiments(r::Vector{Vector{Vector{Float64}}}, granularity::Int, y_list, title, max::Float64)
    avg_r = calc_avg_r_multiple_experiments(r, granularity)
    avg_r_array = [avg_r[i,1,:] for i in 1:size(avg_r)[1]]
    
    x = collect(range(granularity, length(r[1][1]), floor(Int, length(r[1][1])/granularity)))
    
    plot(x, avg_r_array, 
        ylims = (0,10),
        xlabel = "timestep",
        ylabel = "reward",
        title = title, 
        labels=y_list)

    plot!(x, [max for i in x], 
        labels="max")
end

function plot_cumulative_avg_r_multiple_experiments(r::Vector{Vector{Vector{Float64}}}, discount::Float64, labels, title)
    time = length(r[1][1])
    
    avg_r = [[mean([run[i] for run in exp]) for i in 1:time] for exp in r]
    
    disc_vector = [discount^i for i in 0:time-1]
    discounted_avg_r = [elem .* disc_vector for elem in avg_r]
    cumulative_discounted_avg_r = [cumsum(elem) for elem in discounted_avg_r]
    
    x = collect(1:time)
    
    plot(x, cumulative_discounted_avg_r, 
        xlabel = "timestep",
        ylabel = "reward",
        title = title, 
        labels=labels,
        legend=:topleft)
end


function calc_avg_r_multiple_experiments(r::Vector{Vector{Vector{Float64}}}, granularity::Int)
    n_exps = length(r)
    runs = length(r[1])
    time = length(r[1][1])
    @assert time%granularity == 0
    
    points = floor(Int, time/granularity)
    timesteps = zeros(points)
    
    r_avg_across_windows = zeros(n_exps, runs, points)
    for exp in 1:n_exps
        r_exp = r[exp]
        @assert length(r_exp) == runs        
        for run in 1:runs
            r_run = r_exp[run]
            @assert length(r_run) == time
            for i in 1:points
                
                # calc average across window
                en = i*granularity
                st = en-(granularity-1)
                avg_across_window = mean(r_run[st:en])

                r_avg_across_windows[exp,run,i] = avg_across_window
            end
        end
    end
    
    # calc average across runs
    r_avg_across_windows_runs = mean(r_avg_across_windows, dims=2)
    
    return r_avg_across_windows_runs
end

function calc_avg_r_multiple_experiments(r::Vector{Vector{Any}}, granularity::Int)
    n_exps = length(r)
    runs = length(r[1])
    time = length(r[1][1])
    @assert time%granularity == 0
    
    points = floor(Int, time/granularity)
    timesteps = zeros(points)
    
    r_avg_across_windows = zeros(n_exps, runs, points)
    for exp in 1:n_exps
        r_exp = r[exp]
        @assert length(r_exp) == runs        
        for run in 1:runs
            r_run = r_exp[run]
            @assert length(r_run) == time
            for i in 1:points
                
                # calc average across window
                en = i*granularity
                st = en-(granularity-1)
                avg_across_window = mean(r_run[st:en])

                r_avg_across_windows[exp,run,i] = avg_across_window
            end
        end
    end
    
    # calc average across runs
    r_avg_across_windows_runs = mean(r_avg_across_windows, dims=2)
    
    return r_avg_across_windows_runs
end

function plot_avg_r(r::Vector{Vector{Float64}}, granularity::Int, y, expID)
    # calc avg r over range
    @assert length(r[1])%granularity == 0

    points = floor(Int, length(a)/granularity)
    timesteps = zeros(points)
    avg_r = zeros((runs, points))
    for i in 1:points
        en = i*granularity
        st = en-(granularity-1)
        for run in 1:runs
            avg = mean(r[run][st:en])
            timesteps[i] = i*granularity
            avg_r[run,i] = avg
        end
    end

    avg_r_array = [vec(avg_r[i,:]) for i in 1:(size(avg_r)[1])]

    # plot avg r v. timestep
    plot(timesteps, avg_r_array,
        ylims = (0,10),
        ylabel = "reward (avg over "*string(granularity)*" steps)",
        xlabel = "timestep",
        legend = false,
        title = "avg reward, y="*string(y)*" (exp "*expID*", "*string(runs)*" runs)")
end

function plot_proportion_actions_in_list(a::Vector{Vector{SubString{String}}}, a_list::Array{String}, gran::Int, y, expID)
    @assert length(a[1])%gran == 0
    
    valid_a = [[arm in a_list for arm in a[i]] for i in 1:runs]
    
    n = floor(Int, length(a[1])/gran)
    timesteps = zeros(n)
    percent_valid = zeros((runs, n))
    for i in 1:n
        f = i*gran
        s = f-(gran-1)
        timesteps[i] = i*gran
        for run in 1:runs
            percent_valid[run,i] = mean(valid_a[run][s:f])
        end
    end
    
    avg_percent_valid = [mean(percent_valid[:,i]) for i in 1:n]
    sd_percent_valid = [std(percent_valid[:,i]) for i in 1:n]
    
    # plot percent valid actions v. timestep
    plot(timesteps, avg_percent_valid,
        ribbon = sd_percent_valid,
        ylims = (0,1.2),
        ylabel = "% a in "*string(a_list)*" ("*string(gran)*" steps)" ,
        xlabel = "timestep",
        legend = false,
        title = "actions, y="*string(y)*" (exp "*expID*", "*string(runs)*" runs)")
end
                    
function plot_proportion_actions_in_list_rolling(a, a_list::Array{String}, window::Int, y, expID)
    runs = length(a)
    time = length(a[1])
    @assert time/window >= 2
    
    valid_a = [[arm in a_list for arm in a[i]] for i in 1:runs]
    avg_valid_a = [mean([x[i] for x in valid_a]) for i in 1:time]
    sd_valid_a = [std([x[i] for x in valid_a]) for i in 1:time]
    
    avg_moving_window_percent_valid = [i < window ? mean(avg_valid_a[begin:i]) : mean(avg_valid_a[i-window+1:i]) for i in 1:length(avg_valid_a)]
    sd_moving_window_percent_valid = [i < window ? mean(sd_valid_a[begin:i]) : mean(sd_valid_a[i-window+1:i]) for i in 1:length(sd_valid_a)]

    # plot percent valid actions v. timestep
    plot(1:time, avg_moving_window_percent_valid,
#         ribbon = sd_moving_window_percent_valid,
        ylims = (0,1.2),
        ylabel = "% a in list "*string(a_list)*" ("*string(window)*" window)" ,
        xlabel = "timestep",
        legend = false,
        title = "actions, y="*string(y)*" (exp "*expID*", "*string(runs)*" runs)")
end

function plot_actions_in_list_rolling_multiple_experiments(a, a_list::Array{String}, window::Int, labels, title::String)
    exps = length(a)
    runs = length(a[1])
    time = length(a[1][1])
    @assert time/window >= 2
    
    a_prop = Array{Vector{Float64}}(undef, exps)
    for exp in 1:exps
        a_prop[exp] = get_proportion_actions_in_list_rolling(a[exp], window, a_list)
    end
    
    plot(collect(1:time), a_prop,
        ylims = (0,1.),
        xlabel = "timestep",
        ylabel = "avg percent of actions",
        labels = labels,
        title = title*string(a_list))
end
                                        
function get_proportion_actions_in_list_rolling(a, window::Int, a_list::Array{String})
    runs = length(a)
    time = length(a[1])
    @assert time/window >= 2
    
    valid_a = [[arm in a_list for arm in a[i]] for i in 1:runs]
    avg_valid_a = [mean([x[i] for x in valid_a]) for i in 1:time]    
    avg_moving_window_percent_valid = [i < window ? mean(avg_valid_a[begin:i]) : mean(avg_valid_a[i-window+1:i]) for i in 1:length(avg_valid_a)]

    return avg_moving_window_percent_valid
end

function plot_proportion_actions_all(a, actions, window, title)
    avg_percent_action = Array{Array{Float64}}(undef, length(actions))
    for i in 1:length(actions)
        avg_percent_action[i] = get_proportion_actions_in_list_rolling(a, window, [actions[i]])
    end
    labels = reshape(actions, 1, length(actions))
    plot(1:length(a[1]), avg_percent_action,
        ylims = (0,1.0),
        ylabel = "% a ("*string(window)*" step window)" ,
        labels = labels,
        title = title,
        xlabel = "timestep")
end

function plot_proportion_actions_B(a, actions, window, title)
    avg_percent_action = Array{Array{Float64}}(undef, length(actions))
    for i in 1:length(actions)
        avg_percent_action[i] = get_proportion_actions_in_list_rolling(a, window, [actions[i]])
    end
    labels = reshape(actions, 1, length(actions))
    plot(1:length(a[1]), avg_percent_action,
        ylims = (0,0.08),
        ylabel = "% a ("*string(window)*" step window)" ,
        labels = labels,
        title = title,
        xlabel = "timestep")
end

function plot_proportion_high_B(a, gran::Int, y_labels, high_B, b_list, title)
    exps = length(a)
    runs = length(a[1])
    time = length(a[1][1])
    @assert time%gran == 0
        
    any_b = [[[arm in b_list for arm in a[exp][run]] for run in 1:runs] for exp in 1:exps]
    high_b = [[[arm in high_B for arm in a[exp][run]] for run in 1:runs] for exp in 1:exps]

    n = floor(Int, time/gran)
    percent_high_B = zeros((exps, runs, n))
    for exp in 1:exps
        for i in 1:n
            f = i*gran
            s = f-(gran-1)
            for run in 1:runs
                times_any_B = sum(any_b[exp][run][s:f])
                times_high_B = sum(high_b[exp][run][s:f])
                percent_high_B[exp,run,i] = times_high_B/times_any_B
            end
        end
    end
    
    # avg across dim 2, ignoring non elements 
    b_avg = nanmean(percent_high_B, dims=2)    
    b_avg_array = [b_avg[i,1,:] for i in 1:size(b_avg)[1]]
    
    x = collect(range(gran, length(a[1][1]), floor(Int, length(a[1][1])/gran)))
    
    plot(x, b_avg_array,
        ylims = (0,1.),
        labels = y_labels,
        title = title,
        ylabel = "% high B/% any B ("*string(gran)*" window)",
        xlabel = "timestep")
    end

function plot_num_teacher_queries(a, labels, title)
    T = length(a[1][1])
    queries = [[[run[t][1]=='B' for t in 1:T] for run in exp] for exp in a]
    queries_sum = [[sum(run, dims=1)[1] for run in exp] for exp in queries]
    queries_sum_avg = [mean(exp) for exp in queries_sum]
    queries_sum_std = [std(exp) for exp in queries_sum]

    bar(vec(labels), queries_sum_avg,
        yerr = queries_sum_std,
        xlabel = "algorithm",
        ylabel = "teacher queries",
        title = title,
        legend = false)
end

function plot_final_reward_frequency(r, s, runs, trail=100)
    avg = [mean(exp[end-(trail-1):end]) for exp in r]
    max_vec = zeros(size(r))
    for i in 1:size(s)[1]
        max_vec[(i-1)*runs+1:(i)*runs] .= maximum([dot(s[i].u, d) for d in s[i].d])
    end
    scaled = avg./max_vec
    histogram(scaled,
        xlabel= "scaled reward",
        ylabel= "frequency",
        title= "scaled reward of run (averaged over final "*string(trail)*" steps)",
        legend= false)
end

function plot_inferred_state_loss(final_states, s)
    u_err = []
    d_err = []
    ud_err = []
    s_true = repeat(s, inner=size(final_states)[1])
    for i in 1:size(final_states)[1]
        u_diff = s_true[1].u .- final_states[i].u
        append!(u_err, norm(u_diff))
        d_diff = s_true[1].d .- final_states[i].d
        for d in d_diff
            append!(d_err, norm(d)) 
        end
        ud_diff = [dot(s_true[1].u, d) for d in s_true[1].d] .- [dot(final_states[i].u, d) for d in final_states[i].d]
        for ud in ud_diff
            append!(ud_err, norm(ud)) 
        end
    end

    u_avg = mean(u_err)
    u_std = std(u_err)
    d_avg = mean(d_err)
    d_std = std(d_err)
    ud_avg = mean(ud_err)
    ud_std = std(ud_err)

    bar(["item utilities", "expected arm utilities", "arm distributions"], [u_avg, ud_avg, d_avg],
        yerr = [u_std, d_std, ud_std],
        title = "avg L2 loss",
        legend = false)
end
end