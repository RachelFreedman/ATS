# args: test_boolean num_items num_arms utility_granularity arm_granularity num_runs run_length state_index max_depth seed
# takes 45m
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 1572 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 1713 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 3194 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 3607 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 4151 5 1
julia ./experiment_scripts/ATS_infinite.jl false 3 3 0.9 3 3 20 1000 4423 5 1