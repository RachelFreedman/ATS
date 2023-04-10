using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPPolicies, Parameters, Random, Plots, LinearAlgebra, POMDPTools, BasicPOMCP, D3Trees, GridInterpolations, POMDPModels, Combinatorics, Dates, Serialization, ParticleFilters

FILTER = ARGS[1] == "true"

@with_kw struct MyParameters
    filter = FILTER
    N::Int = parse(Int64, ARGS[2])           # size of item set
    K::Int = parse(Int64, ARGS[3])           # size of arm set
    M::Int = parse(Int64, ARGS[4])           # size of beta set
    umax::Real = 10                          # max utility
    u_grain::Int = parse(Int64, ARGS[5])     # granularity of utility approximation
    d_grain::Int = parse(Int64, ARGS[6])     # granularity of arm distribution approximation
    exp_steps::Int = 100
    beta:: Array{Float64} = M == 3 ? [0., 0.01, 50.] : [0., 0.01, 1., 50.]  # teacher beta values
end

params = MyParameters()
println(string(params))

function no_duplicates(a::Array)
    for i in 1:length(a)-1
        for j in i+1:length(a)
            if a[i] == a[j]
                return false
            end
        end
    end
    return true
end

struct State
    t::Int                    # timesteps remaining
    u::Array{Float64}         # list of N utility values for N items
    d::Array{Array{Float64}}  # list of K arm distributions, each assigning probabilities to N items
    b::Array{Float64}         # list of M beta values
end

# space of utility functions
@time begin
    umin = 0
    grid_coor = fill(range(umin,params.umax,length=params.u_grain), params.N)
    U = RectangleGrid(grid_coor...)
end

if FILTER
    U = filter(x -> no_duplicates(x), collect(U))
end

@assert length(U[1]) == params.N
println("generated "*string(length(U))*" utilities (each length "*string(length(U[1]))*" items)")

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
                println("ERROR: lst "*string(lst)*" has type "*string(typeof(lst[1]))*". Must be Float64.")
            end
            prepend!(lst, float(k))
        end
        out = vcat(out, subsolution)
    end
    return out
end

# space of arm distributions
@time begin
    coor = collect(range(0.,1.,length=params.d_grain))    
    simplex_list = generate_probability_distributions(params.N, coor)
    D_tuples = vec(collect(Base.product(fill(simplex_list, params.K)...)))
    D = [collect(d) for d in D_tuples]
end

if FILTER
    D = filter(x -> no_duplicates(x), D)
end

@assert length(D[1]) == params.K
@assert length(D[1][1]) == params.N
println(string("generated "*string(length(D))*" arm distribution sets (each shape "*string(length(D[1]))*" arms x "*string(length(D[1][1]))*" items)"))

# beta values
B = [params.beta]

# each beta value set must be length M
@assert length(B[1]) == params.M
println(string("generated "*string(length(B))*" beta value sets (each length "*string(length(B[1]))*" teachers)"))

# State space
@time begin    
    S = [[State(t,u,d,b) for u in U, d in D, b in B, t in params.exp_steps:-1:1]...,]
end

# absorbing state
final = State(0, zeros(params.N), [zeros(params.N) for _ in 1:params.K], zeros(params.M))
push!(S, final)

# initial states
S_init = S[1:length(U)*length(D)*length(B)]

println("generated "*string(length(S))*" states, "*string(length(S_init))*" of which are initial states")

## List "interesting" state IDs

function interesting(s::State)
    e = [dot(s.u, di) for di in s.d]
    # C3 > 0
    order = e[3] > 0
    # C1 > all other arms
    for i in 2:length(e)
        if e[1] <= e[i]
            order = false
        end
    end
    # C2 > all subsequent arms
    for i in 3:length(e)
        if e[2] <= e[i]
            order = false
        end
    end
    diff = s.d[1] != s.d[2] != s.d[3]
    stoch = all([s.d[i][j]!=1. for i in 1:3, j in 1:3])
    return order && stoch && diff
end

to_print = false
indices = []
for i in 1:length(S_init)
    if interesting(S_init[i])
        if to_print
            println("index ", i)
            println(S_init[i])
        end
        push!(indices, i)
    end
end

println("\n")
println("Total States: "*string(length(S_init)))
println("Interesting States: "*string(length(indices)))
println(indices)