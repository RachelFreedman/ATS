# args: test_boolean num_items num_arms utility_granularity arm_granularity num_runs run_length state_index max_depth seed
# takes 45m
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 181 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 207 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 398 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 442 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 521 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 25 1000 564 5 1