using Serialization, ParticleFilters, JSON

struct State
    t::Int                    # timesteps remaining
    u::Array{Float64}         # list of N utility values for N items
    d::Array{Array{Float64}}  # list of K arm distributions, each assigning probabilities to N items
    b::Array{Float64}         # list of M beta values
end

# ai: active infinite
ids_ai = ["active_infinite_23422_131424", "active_infinite_23419_153324", "active_infinite_23419_154231", "active_infinite_23419_155139", "active_infinite_23419_160549", "active_infinite_23419_161648", "active_infinite_23419_162359", "active_infinite_23419_163115", "active_infinite_23419_16401", "active_infinite_23419_164925", "active_infinite_23419_17049", "active_infinite_23419_171313", "active_infinite_23419_172230", "active_infinite_23419_172929", "active_infinite_23419_17394", "active_infinite_23419_174759", "active_infinite_23419_180013", "active_infinite_23419_181126", "active_infinite_23419_182029", "active_infinite_23419_183010"]

# af: active finite
ids_af = ["active_finite_23419_214856", "active_finite_23419_22017", "active_finite_23419_22124", "active_finite_23419_222216", "active_finite_23419_223627", "active_finite_23419_22505", "active_finite_23419_230052", "active_finite_23419_230927", "active_finite_23419_232011", "active_finite_23419_233116", "active_finite_23419_23426", "active_finite_23419_235054", "active_finite_23420_000135", "active_finite_23420_001238", "active_finite_23420_002515", "active_finite_23420_004012", "active_finite_23420_00540", "active_finite_23420_010552", "active_finite_23420_011651", "active_finite_23420_012749"]

# pi: passive infinite
ids_pi = ["passive_infinite_23419_183940", "passive_infinite_23419_184948", "passive_infinite_23419_18573", "passive_infinite_23419_191144", "passive_infinite_23419_193217", "passive_infinite_23419_194023", "passive_infinite_23419_194838", "passive_infinite_23419_195741", "passive_infinite_23419_200444", "passive_infinite_23419_201343", "passive_infinite_23419_202155", "passive_infinite_23419_203025", "passive_infinite_23419_203934", "passive_infinite_23419_205227", "passive_infinite_23419_205917", "passive_infinite_23419_210814", "passive_infinite_23419_211748", "passive_infinite_23419_21258", "passive_infinite_23419_213346", "passive_infinite_23419_214029"]

# pf: passive finite
ids_pf = ["passive_finite_23422_141255", "passive_finite_23422_142713", "passive_finite_23422_143917", "passive_finite_23422_145237", "passive_finite_23422_150344", "passive_finite_23422_151540", "passive_finite_23422_152710", "passive_finite_23422_153914", "passive_finite_23422_155019", "passive_finite_23422_16003", "passive_finite_23422_161248", "passive_finite_23422_162128", "passive_finite_23422_163156", "passive_finite_23422_164250", "passive_finite_23422_165540", "passive_finite_23422_17041", "passive_finite_23422_17148", "passive_finite_23422_172518", "passive_finite_23422_173548", "passive_finite_23422_174423"]

function write_to_json(b::Dict{String, Matrix{Array{ParticleCollection{State}}}}; runs=25, test=false)
    for id in keys(b)
        id_dict = Vector{Vector{Dict{State, Float64}}}(undef, runs)
        for i in 1:runs
            result = map(ParticleFilters.probdict, b[id][i])
            id_dict[i] = result
        end
        json_str = JSON.json(id_dict)
        if test
            id = "test"
        end
        open("data/beliefs/"*id*".json", "w") do f 
            write(f, json_str) 
        end
        println("wrote "*id*".json")
    end
end

function write_to_json(b::Matrix{Array{ParticleCollection{State}}}, id::String; runs=25, test=false)
    id_dict = Vector{Vector{Dict{State, Float64}}}(undef, runs)
    for i in 1:runs
        result = map(ParticleFilters.probdict, b[i])
        id_dict[i] = result
    end
    json_str = JSON.json(id_dict)
    if test
        id = "test"
    end
    open("data/beliefs/"*id*".json", "w") do f 
        write(f, json_str) 
    end
    println("wrote "*id*".json")
end

# println("beginning test")

# for id in ["active_infinite_23422_131424"]
#     @time begin
#         b = deserialize(open("beliefs/"*id*"_belief.txt", "r"))
#         println("deserialized ", id)
#     end
#     @time begin
#         write_to_json(b, id, test=true)
#     end
# end

println("beginning deserialization")

for id in ids_ai
    @time begin
        b = deserialize(open("beliefs/"*id*"_belief.txt", "r"))
        println("deserialized ", id)
    end
    @time begin
        write_to_json(b, id)
    end
end

for id in ids_af
    @time begin
        b = deserialize(open("beliefs/"*id*"_belief.txt", "r"))
        println("deserialized ", id)
    end
    @time begin
        write_to_json(b, id)
    end
end

for id in ids_pi
    @time begin
        b = deserialize(open("beliefs/"*id*"_belief.txt", "r"))
        println("deserialized ", id)
    end
    @time begin
        write_to_json(b, id)
    end
end

for id in ids_pf
    @time begin
        b = deserialize(open("beliefs/"*id*"_belief.txt", "r"))
        println("deserialized ", id)
    end
    @time begin
        write_to_json(b, id)
    end
end

# @time begin
#     beliefs_pf = Dict{String, Matrix{Array{ParticleCollection{State}}}}()
#     for id in ids_pf
#         beliefs_pf[id] = deserialize(open("../beliefs/"*id*"_belief.txt", "r"))
#         println("deserialized ", id)
#     end
#     write_to_json(beliefs_pf)
# end
    