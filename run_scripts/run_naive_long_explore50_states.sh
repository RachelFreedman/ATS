# args: test_boolean num_items num_arms utility_granularity arm_granularity num_runs run_length state_index explore_length teacher seed
# takes 5m
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 1572 50 2 1
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 1713 50 2 1
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 3194 50 2 1
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 3607 50 2 1
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 4151 50 2 1
julia ./experiment_scripts/naive.jl false 3 3 3 3 25 1000 4423 50 2 1