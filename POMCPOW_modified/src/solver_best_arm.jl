# include("./POMCPOW.jl")
# using .POMCPOW: IdentityUpdater

### BestArmPolicy ###
"""
BestArmPolicy{RNG<:AbstractRNG, A<:Array, P<:Union{POMDP,MDP}, U<:Updater}
a generic policy that randomly samples an action from the provided action set A.

Constructor:

    `BestArmPolicy(problem::Union{POMDP,MDP},
             actions::Array;
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `A::Array` set of valid actions, must be a subset of actions(P)
- `problem::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `NothingUpdater` in the above constructor)
"""
mutable struct BestArmPolicy{RNG<:AbstractRNG, A<:Array, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    actions::A
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
BestArmPolicy(problem::Union{POMDP,MDP},
             actions::Array;
             rng=Random.GLOBAL_RNG,
             updater=IdentityUpdater()) = BestArmPolicy(rng, actions, problem, updater)

function action(policy::BestArmPolicy, b)
    s = rand(policy.rng, b)
    println("sampled state: ", s)
    best_arm = argmax([dot(s.u, d) for d in s.d])
    println("best arm according to sample: ", best_arm)
    a = policy.actions[best_arm]
    return a
end

## convenience functions ##
updater(policy::BestArmPolicy) = policy.updater


"""
solver that produces a constrained random policy
"""
mutable struct BestArmSolver <: Solver
    actions::Array
    rng::AbstractRNG
end
BestArmSolver(actions::Array;rng=Random.GLOBAL_RNG) = BestArmSolver(actions, rng)

solve(solver::BestArmSolver, problem::Union{POMDP,MDP}) = BestArmPolicy(solver.rng, solver.actions, problem, IdentityUpdater())

mutable struct IdentityUpdater <: Updater end

initialize_belief(::IdentityUpdater, b::StateBelief) = b

function update(bu::IdentityUpdater, b::StateBelief, a, o) 
    return b
end

POMDPs.update(bu::IdentityUpdater, b::Any, a, o) = update(bu, initialize_belief(bu, b), a, o)
BasicPOMCP.extract_belief(::IdentityUpdater, node::BeliefNode) = belief(node)