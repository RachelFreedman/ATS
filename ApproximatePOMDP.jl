# ARGS = [N, K, discount, u_grain, d_grain, exp_iters, exp_steps]

using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPPolicies, Parameters, Random, Plots, LinearAlgebra, POMDPTools, BasicPOMCP, D3Trees, GridInterpolations, POMCPOW, POMDPModels, Combinatorics, Dates, Serialization, ParticleFilters

expID = Dates.format(Dates.now(), "yymd_HHMMS")

function log(s::String)
    s_time = Dates.format(Dates.now(), "HH:MM:SS\t")*s*"\n"
    open("./logs/"*expID*".txt", "a") do file
        write(file, s_time)
    end
    print(s_time)
end

log("Running experiment with ID "*expID)

@with_kw struct MyParameters
    N::Int = parse(Int64, ARGS[1])           # size of item set
    K::Int = parse(Int64, ARGS[2])           # size of arm set
    M::Int = 2                               # size of beta set
    y::Float64 = parse(Float64, ARGS[3])     # discount factor
    umax::Real = 10                          # max utility
    u_grain::Int = parse(Int64, ARGS[4])     # granularity of utility approximation
    d_grain::Int = parse(Int64, ARGS[5])     # granularity of arm distribution approximation
    beta:: Array{Float64} = [0.01, 10.0]     # teacher beta values
    exp_iters::Int = parse(Int64, ARGS[6])   # number of rollouts to run
    exp_steps::Int = parse(Int64, ARGS[7])   # number of timesteps per rollout
end

params = MyParameters()
log(string(params))

struct State
    u::Array{Float64}         # list of N utility values for N items
    d::Array{Array{Float64}}  # list of K arm distributions, each assigning probabilities to N items
    b::Array{Float64}         # list of M beta values
end

# space of utility functions
umin = 0
grid_coor = fill(range(umin,params.umax,length=params.u_grain), params.N)
U = RectangleGrid(grid_coor...)

@assert length(U[1]) == params.N
log("generated "*string(length(U))*" utilities (each length "*string(length(U[1]))*" items)")

function generate_probability_distributions(N::Int, coor::Array{Float64}, S::Float64=1.0)
    if S == 0
        return [[0. for _ in 1:N]]
    end
    if N == 1
        return [[float(S)]]
    end
    out = []
    range = coor[1:findall(x->isapprox(x,S,atol=1e-15), coor)[1]]
    for k in range
        subsolution = generate_probability_distributions(N-1, coor, S-k)
        for lst in subsolution
            if typeof(lst[1]) != Float64
                log("ERROR: lst "*string(lst)*" has type "*string(typeof(lst[1]))*". Must be Float64.")
            end
            prepend!(lst, float(k))
        end
        out = vcat(out, subsolution)
    end
    return out
end

# space of arm distributions
coor = collect(range(0.,1.,length=params.d_grain))    
simplex_list = generate_probability_distributions(params.N, coor)
D_tuples = vec(collect(Base.product(fill(simplex_list, params.K)...)))
D = [collect(d) for d in D_tuples]

@assert length(D[1]) == params.K
@assert length(D[1][1]) == params.N
log("generated "*string(length(D))*" arm distribution sets (each shape "*string(length(D[1]))*" arms x "*string(length(D[1][1]))*" items)")

# beta values
B = [params.beta]

# each beta value set must be length M
@assert length(B[1]) == params.M
log("generated "*string(length(B))*" beta value sets (each length "*string(length(B[1]))*" teachers)")

# State space
S = [[State(u,d,b) for u in U, d in D, b in B]...,]

log("generated "*string(length(S))*" states")

# Action space - actions are arm choices (K) or beta selections (M)
struct Action
    name::String      # valid names are {B,C} + index
    isBeta::Bool      # true if 'B' action, false if 'C' action
    index::Integer    # index of beta (if 'B' action) or arm choice (if 'C' action)
end

A = Array{Action}(undef, params.K+params.M)
for i in 1:params.K+params.M
    if i <= params.K
        A[i] = Action("C"*string(i), false, i)
    else
        A[i] = Action("B"*string(i-params.K), true, i-params.K)
    end
end
log("generated "*string(length(A))*" actions")

# Transition function
function T(s::State, a::Action)
    return SparseCat([s], [1.0])    # categorical distribution
end
log("generated transition function")

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

P = [[Preference(i0,i1,label) for i0 in I, i1 in I, label in [0,1]]...,]

# observation space
struct Observation
    isItem::Bool    # true if item returned, false otherwise
    i::Int          # item, if item returned
    p::Preference   # preference, if preference returned
end

invalid_i = -1
invalid_p = Preference(-1,-1,-1)
I_obs = [Observation(true, i, invalid_p) for i in I]
P_obs = [Observation(false, invalid_i, p) for p in P]
omega = union(I_obs, P_obs)

log("generated "*string(length(omega))*" observations")

# unnormalized query profile (likelihood of querying 1,1; 2,1; 3,1; ... ; N,1; 1,2; 2,2; ... ; N,N)
Q = ones(params.N*params.N)

# preference probability (expected preference, or probability that preference=1)
function Pr(p::Preference, s::State, b::Float64)
    prob_pref_1 = exp(Float64(b)*s.u[p.i1])/(exp(Float64(b)*s.u[p.i1])+exp(Float64(b)*s.u[p.i0]))
    if p.label == 1
        return prob_pref_1
    else
        return 1.0-prob_pref_1
    end
end

function O(s::State, a::Action, sp::State)
    # if B action, obs in P_obs
    if a.isBeta
        prob_of_pref = [Pr(o.p, s, s.b[a.index]) for o in P_obs]
        prob_of_query = vcat(Q,Q)   # doubled because each query appears once for each label
        
        # weight by querying profile to get dist
        dist = [prob_of_pref[i]*prob_of_query[i] for i in 1:length(prob_of_pref)]
        normalized_dist = dist/sum(dist)        
        return SparseCat(P_obs, normalized_dist)
    # if C action, obs in I_obs
    else
        return SparseCat(I_obs, s.d[a.index])
    end
end

log("generated observation function")
    
# define POMDP
abstract type MyPOMDP <: POMDP{State, Action, Observation} end
pomdp = QuickPOMDP(MyPOMDP,
    states       = S,
    actions      = A,
    observations = omega,
    transition   = T,
    observation  = O,
    reward       = R,
    discount     = params.y,
    initialstate = S);

log("created POMDP")

# solve POMDP with POMCPOW
solver = POMCPOWSolver()
planner = solve(solver, pomdp);
log("solved POMDP")

# save serialized planner
open("./policies/"*expID*"_policy.txt", "w") do file
    serialize(file, planner)
end

log("saved policy to "*"./policies/"*expID*"_policy.txt")

# simulate rollouts
steps = params.exp_steps
iters = params.exp_iters

# POMCPOW rollouts
# hardcoded initial states
initial_states = [State([0.0, 8.0, 10.0], Array{Float64}[[0.0, 0.0, 1.0], [0.6, 0.0, 0.4], [1.0, 0.0, 0.0]], [0.01, 10.0]) for i in 1:iters]
POMCPOW_R = Array{Float64}(undef, iters)
beliefs = Array{Array{ParticleFilters.ParticleCollection{State}}}(undef, (iters, steps))
for iter in 1:iters
    log("logging POMCPOW simulation "*string(iter)*" to "*"./sims/"*expID*"_run"*string(iter)*".txt")
    t = 1
    r_accum = 0.
    beliefs_iter = Array{ParticleFilters.ParticleCollection{State}}(undef, steps)
    for (s, a, o, r, b) in stepthrough(pomdp, planner, updater(planner), Uniform(S), initial_states[iter], "s,a,o,r,b", max_steps=steps)
        r_accum = r_accum + r
        beliefs_iter[t] = b
        if t == 1
            open("./sims/"*expID*"_run"*string(iter)*".txt", "w") do file
                write(file, string(s))
            end
#             initial_states[iter] = s
        end
        if a.isBeta
            msg = "\n"*string(t)*",B,"*a.name*",(i"*string(o.p.i0)*"-i"*string(o.p.i1)*";"*string(o.p.label)*"),"*string(r)
        else
            msg = "\n"*string(t)*",C,"*a.name*",i"*string(o.i)*","*string(r)
        end
        open("./sims/"*expID*"_run"*string(iter)*".txt", "a") do file
            write(file, msg)
        end
        t = t + 1
    end
    beliefs[iter] = beliefs_iter
    POMCPOW_R[iter] = r_accum
end
    
log("ran "*string(iters)*" POMCPOW rollouts for "*string(steps)*" timesteps each")
log("POMCPOW R: "*string(POMCPOW_R))

open("./beliefs/"*expID*"_belief.txt", "w") do file
    serialize(file, beliefs)
end

log("saved beliefs to "*"./beliefs/"*expID*"_belief.txt")

# random rollouts
prior = Uniform(S)
sim = RolloutSimulator(max_steps=steps)

random_R = zeros(iters)
for iter in 1:iters
    # use the same initial states as the POMCPOW runs
    initial_state = initial_states[iter]
    up = updater(RandomPolicy(pomdp))
    result = simulate(sim, pomdp, RandomPolicy(pomdp), up, prior, initial_state)
    random_R[iter] = result
end

log("ran "*string(iters)*" random rollouts for "*string(steps)*" timesteps each")
log("Random R: "*string(random_R))

# calculate maximum possible reward
max_R = zeros(iters)

for iter in 1:iters
    # use the same initial states as the POMCPOW runs
    initial_state = initial_states[iter]
    max_R[iter] = maximum([dot(initial_state.u, initial_state.d[i]) for i in 1:params.K])*steps
end

log("Max R: "*string(max_R))

# plot figure
fig = plot(1:iters, [random_R,POMCPOW_R], 
    seriestype = :scatter, 
    label=["random" "POMCP"], 
    xticks = 0:1:iters,
    xlabel = "run",
    ylabel = "reward (" * string(steps) * " timesteps)",
    ylims = (0,maximum(POMCPOW_R)*1.2)
)
savefig(fig,"./plots/reward_ID"*string(expID)*"_step"*string(steps)*"_roll"*string(iters)*".png")
log("saved fig to ./plots/reward_ID"*string(expID)*"_step"*string(steps)*"_roll"*string(iters)*".png")