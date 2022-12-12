using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPPolicies, Parameters, Random, Plots, LinearAlgebra, Serialization, StatsBase, POMDPTools, BasicPOMCP, D3Trees, GridInterpolations, POMCPOW, POMDPModels, Combinatorics, Dates, CSV, ParticleFilters

function log(s::String)
    s_time = Dates.format(Dates.now(), "HH:MM:SS\t") * s * "\n"
    open("./logs/" * expID * ".txt", "a") do file
        write(file, s_time)
    end
    print(s_time)
end

## NAIVE BASELINE ##
# use "naive" algorithm

exp_name = "base_naive_"
expID = exp_name * Dates.format(Dates.now(), "yymd_HHMMS")
log("Running experiment with ID " * expID)

@with_kw struct MyParameters
    N::Int = parse(Int64, ARGS[1])           # size of item set
    K::Int = parse(Int64, ARGS[2])           # size of arm set
    M::Int = 3                               # size of beta set
    y::Float64 = parse(Float64, ARGS[3])     # discount factor
    umax::Real = 10                          # max utility
    u_grain::Int = parse(Int64, ARGS[4])     # granularity of utility approximation
    d_grain::Int = parse(Int64, ARGS[5])     # granularity of arm distribution approximation
    beta::Array{Float64} = [0., 0.01, 50.]   # teacher beta values
    exp_iters::Int = parse(Int64, ARGS[6])   # number of rollouts to run
    exp_steps::Int = parse(Int64, ARGS[7])   # number of timesteps per rollout
    s_index::Int = parse(Int64, ARGS[8])     # index of true state
end
params = MyParameters()
log(string(params))

# baseline-specific parameters
t_explore = 100
log("will explore for first "*string(t_explore)*" timesteps")
teacher = 3
log("will estimate based on feedback from teacher "*string(teacher)*" with beta "*string(params.beta[teacher]))

struct State
    u::Array{Float64}         # list of N utility values for N items
    d::Array{Array{Float64}}  # list of K arm distributions, each assigning probabilities to N items
    b::Array{Float64}         # list of M beta values
end

# space of utility functions
umin = 0
grid_coor = fill(range(umin, params.umax, length=params.u_grain), params.N)
U = RectangleGrid(grid_coor...)

@assert length(U[1]) == params.N
log("generated " * string(length(U)) * " utilities (each length " * string(length(U[1])) * " items)")

function generate_probability_distributions(N::Int, coor::Array{Float64}, S::Float64=1.0)
    if S == 0
        return [[0.0 for _ in 1:N]]
    end
    if N == 1
        return [[float(S)]]
    end
    out = []
    range = coor[1:findall(x -> isapprox(x, S, atol=1e-15), coor)[1]]
    for k in range
        subsolution = generate_probability_distributions(N - 1, coor, S - k)
        for lst in subsolution
            if typeof(lst[1]) != Float64
                log("ERROR: lst " * string(lst) * " has type " * string(typeof(lst[1])) * ". Must be Float64.")
            end
            prepend!(lst, float(k))
        end
        out = vcat(out, subsolution)
    end
    return out
end

# space of arm distributions
coor = collect(range(0.0, 1.0, length=params.d_grain))
simplex_list = generate_probability_distributions(params.N, coor)
D_tuples = vec(collect(Base.product(fill(simplex_list, params.K)...)))
D = [collect(d) for d in D_tuples]

@assert length(D[1]) == params.K
@assert length(D[1][1]) == params.N
log("generated " * string(length(D)) * " arm distribution sets (each shape " * string(length(D[1])) * " arms x " * string(length(D[1][1])) * " items)")

# beta values
B = [params.beta]

# each beta value set must be length M
@assert length(B[1]) == params.M
log("generated " * string(length(B)) * " beta value sets (each length " * string(length(B[1])) * " teachers)")

# State space
S = [[State(u, d, b) for u in U, d in D, b in B]...,]

log("generated " * string(length(S)) * " states")

# Action space - actions are arm choices (K) or beta selections (M)
struct Action
    name::String      # valid names are {B,C} + index
    isBeta::Bool      # true if 'B' action, false if 'C' action
    index::Integer    # index of beta (if 'B' action) or arm choice (if 'C' action)
end

A = Array{Action}(undef, params.K + params.M)
for i in 1:params.K+params.M
    if i <= params.K
        A[i] = Action("C" * string(i), false, i)
    else
        A[i] = Action("B" * string(i - params.K), true, i - params.K)
    end
end
log("generated " * string(length(A)) * " actions")

# Reward function
function R(s::State, a::Action)
    # if beta selected, return 0
    if a.isBeta
        return 0
        # if arm pulled, return that arm's avg utility
    else
        utilities = s.u
        arm_dist = s.d[a.index]
        return dot(utilities, arm_dist)
    end
end
log("generated reward function")

# item space
I = 1:params.N

# preference space
struct Preference
    i0::Int    # first item to compare, in {1,2,...,N}
    i1::Int    # second item to compare, in {1,2,...,N}
    label::Int # feedback label, in {0,1}
end

P = [[Preference(i0, i1, label) for i0 in I, i1 in I, label in [0, 1]]...,]

# observation space
struct Observation
    isItem::Bool    # true if item returned, false otherwise
    i::Int          # item, if item returned
    p::Preference   # preference, if preference returned
end

invalid_i = -1
invalid_p = Preference(-1, -1, -1)
I_obs = [Observation(true, i, invalid_p) for i in I]
P_obs = [Observation(false, invalid_i, p) for p in P]
omega = union(I_obs, P_obs)

log("generated " * string(length(omega)) * " observations")

# unnormalized query profile (likelihood of querying 1,1; 2,1; 3,1; ... ; N,1; 1,2; 2,2; ... ; N,N)
Q = ones(params.N * params.N)

# preference probability (expected preference, or probability that preference=1)
function Pr(p::Preference, s::State, b::Float64)
    prob_pref_1 = exp(Float64(b) * s.u[p.i1]) / (exp(Float64(b) * s.u[p.i1]) + exp(Float64(b) * s.u[p.i0]))
    if p.label == 1
        return prob_pref_1
    else
        return 1.0 - prob_pref_1
    end
end

function O(s::State, a::Action, sp::State)
    # if B action, obs in P_obs
    if a.isBeta
        prob_of_pref = [Pr(o.p, s, s.b[a.index]) for o in P_obs]
        prob_of_query = vcat(Q, Q)   # doubled because each query appears once for each label

        # weight by querying profile to get dist
        dist = [prob_of_pref[i] * prob_of_query[i] for i in 1:length(prob_of_pref)]
        normalized_dist = dist / sum(dist)
        return SparseCat(P_obs, normalized_dist)
        # if C action, obs in I_obs
    else
        return SparseCat(I_obs, s.d[a.index])
    end
end

log("generated observation function")

function select_arm(K)
    return rand(1:K)
end

function select_teacher(M)
    # always query teacher 2 to use mid-range beta
    return 2
end

function select_query(N)
    return sort(sample(1:N, 2, replace=false))
end

function query_teacher(m, i1, i2, s)
    b = s.b[m]
    p1 = exp(Float64(b) * s.u[i1]) / (exp(Float64(b) * s.u[i1]) + exp(Float64(b) * s.u[i2]))
    return sample(0:1, Weights([p1, 1-p1]))
end
    
function pull_arm(k, s)
    return sample(1:params.N, Weights(s.d[k]))
end     

function get_pref(b)
    query = sort(sample(1:params.N, 2, replace=false))
    i1 = query[1]
    i2 = query[2]
    p1 = exp(Float64(b) * params.u[i1]) / (exp(Float64(b) * params.u[i1]) + exp(Float64(b) * params.u[i2]))
    label = sample(0:1, Weights([p1, 1-p1]))
    return Preference(i1, i2, label)
end

# Naive state estimation

function est_P(a, o, M, N)
    teach_prefs = zeros(M, N, N)
    teach_pulls = zeros(M, N, N)
    for s in 1:length(a)
        if a[s].isBeta
            index = a[s].index
            i0, i1, label = o[s].p.i0, o[s].p.i1, o[s].p.label
            teach_pulls[index, i0, i1] = teach_pulls[index, i0, i1] + 1
            teach_prefs[index, i0, i1] = teach_prefs[index, i0, i1] + label
        end
    end

    P_hat = zeros(M, N, N)
    for index in 1:M
        for i0 in 1:N-1
            for i1 in i0+1:N
                if teach_prefs[index, i0, i1] == 0
                    println("WARNING: teacher "*string(index)*" never prefers item "*string(i1)*" to item "*string(i0)*", so setting preference probability at "*string(eps))
                    P_hat[index, i0, i1] = eps
                else
                    P_hat[index, i0, i1] = teach_prefs[index, i0, i1]/teach_pulls[index, i0, i1]
                end
                P_hat[index, i1, i0] = 1-P_hat[index, i0, i1]
            end
        end
    end

    for i in 1:N
        for m in 1:M
            P_hat[m, i,i] = 0.5
        end
    end
    
    return P_hat
end

function calc_deltas(P_hat, b, N, t)
    deltas = zeros(N, N)
    for i in 1:N
        for j in 1:N
            deltas[j,i] = calc_delta(P_hat[t,i,j],b[t])
            deltas[i,j] = -deltas[j,i]
        end
    end
    return deltas
end

function calc_delta(p, b)
    return (-1/b)*Base.log((1/p)-1)
end

function est_U(deltas, umax, N)
    rnge = maximum(deltas)
    result = findall(x->x==rnge, deltas)[1]
    min_i = result[1][1]
    max_i = result[2][1]
    true_vals = zeros(N)
    for i in 1:N
        val = deltas[max_i,i]
        true_vals[i] = -val*(umax/rnge)
    end
    
    return true_vals
end

function estimate_u(a, o, teacher, M, N, b, umax)
    P_hat = est_P(a, o, M, N)
    deltas = calc_deltas(P_hat, b, N, teacher)
    U_hat = est_U(deltas, umax, N)
    return U_hat
end

function estimate_d(a, o, K, N)
    items_returned = zeros((K, N))
    for s in 1:length(a)
        if !a[s].isBeta
            items_returned[a[s].index, o[s].i] = items_returned[a[s].index, o[s].i] + 1
        end
    end

    D_hat = []
    for row_index in 1:size(items_returned, 1)
        row = items_returned[row_index,:]
        push!(D_hat, row/sum(row))
    end

    return D_hat
end

# calc expected U(arm)
function calc_max_arm(u, d)
    max_val = -999999999
    max_arm = -999999999
    
    for i in 1:length(d)
        val = dot(u, d[i])
        if val > max_val
            max_val = val
            max_arm = i
        end
    end
    
    return max_arm, max_val
end

# Execute naive policy: query specified teacher and pull arms randomly for t_b timesteps, then query argmax arm for remaining timesteps

true_state = S[params.s_index]
log("true state "*string(true_state))

as = []
os = []
rs = []

eps = 0.0000001
random_R = zeros(params.exp_iters)
for iter in 1:params.exp_iters
    log("logging naive policy simulation "*string(iter)*" to "*"./sims/"*expID*"_run"*string(iter)*".txt")
    open("./sims/"*expID*"_run"*string(iter)*".txt", "w") do file
        write(file, string(true_state))
    end
    r_accum = 0.
    for t in 1:t_explore
        msg = ""
        if rand(Bool)
            # select arm
            action = select_arm(params.K)
            a = Action("C"*string(action), false, action)
            
            # pull arm
            item = pull_arm(a.index, true_state)
            o = Observation(true, item, invalid_p)
            r = R(true_state, a)
            r_accum = r_accum + r
            
            push!(as, a)
            push!(os, o)
            push!(rs, r)
            msg = "\n"*string(t)*",C,"*a.name*",i"*string(o.i)*","*string(r)
        else
            # select teacher
            action = select_teacher(params.M)
            a = Action("B"*string(action), true, action)
            
            # query teacher
            q = select_query(params.N)
            label = query_teacher(a.index, q[1], q[2], true_state)
            p = Preference(q[1], q[2], label)
            o = Observation(false, invalid_i, p)
            r = R(true_state, a)
            r_accum = r_accum + r
            
            push!(as, a)
            push!(os, o)
            push!(rs, r)
            msg = "\n"*string(t)*",B,"*a.name*",(i"*string(o.p.i0)*"-i"*string(o.p.i1)*";"*string(o.p.label)*"),"*string(r)
        end
        
        open("./sims/"*expID*"_run"*string(iter)*".txt", "a") do file
            write(file, msg)
        end
    end
    
    log("estimating U using teacher "*string(teacher)*" with beta "*string(params.beta[teacher]))
    
    u_est = estimate_u(as, os, teacher, params.M, params.N, params.beta, params.umax)
    d_est = estimate_d(as, os, params.K, params.N)
    max_a, max_val = calc_max_arm(u_est, d_est)
    
    log("Estimated U: "*string(u_est))
    log("Estimated D: "*string(d_est))
    log("given U and D estimates, highest-reward arm is arm "*string(max_a)*" with reward "*string(max_val))
    
    a = Action("C"*string(max_a), false, max_a)
    for t in t_explore+1:params.exp_steps
        item = pull_arm(a.index, true_state)
        o = Observation(true, item, invalid_p)
        r = R(true_state, a)
        r_accum = r_accum + r

        msg = "\n"*string(t)*",C,"*a.name*",i"*string(o.i)*","*string(r)
        open("./sims/"*expID*"_run"*string(iter)*".txt", "a") do file
            write(file, msg)
        end
    end
    random_R[iter] = r_accum
end

log("ran "*string(params.exp_iters)*" naive policy rollouts for "*string(params.exp_steps)*" timesteps each")
log("Naive R: "*string(random_R))
