### ConstrainedRandomPolicy ###
"""
    ConstrainedRandomPolicy{RNG<:AbstractRNG, A<:Array, P<:Union{POMDP,MDP}, U<:Updater}
a generic policy that randomly samples an action from the provided action set A.

Constructor:

    `ConstrainedRandomPolicy(problem::Union{POMDP,MDP},
             actions::Array;
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `A::Array` set of valid actions, must be a subset of actions(P)
- `problem::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `NothingUpdater` in the above constructor)
"""
mutable struct ConstrainedRandomPolicy{RNG<:AbstractRNG, A<:Array, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    actions::A
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
ConstrainedRandomPolicy(problem::Union{POMDP,MDP},
             actions::Array;
             rng=Random.GLOBAL_RNG,
             updater=NothingUpdater()) = ConstrainedRandomPolicy(rng, actions, problem, updater)

## policy execution ##
function action(policy::ConstrainedRandomPolicy, s)
    return rand(policy.rng, policy.actions)
end

function action(policy::ConstrainedRandomPolicy, b::Nothing)
    return rand(policy.rng, policy.actions)
end

## convenience functions ##
updater(policy::ConstrainedRandomPolicy) = policy.updater


"""
solver that produces a constrained random policy
"""
mutable struct ConstrainedRandomSolver <: Solver
    actions::Array
    rng::AbstractRNG
end
ConstrainedRandomSolver(actions::Array;rng=Random.GLOBAL_RNG) = ConstrainedRandomSolver(actions, rng)

solve(solver::ConstrainedRandomSolver, problem::Union{POMDP,MDP}) = ConstrainedRandomPolicy(solver.rng, solver.actions, problem, NothingUpdater())