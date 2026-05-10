#!/bin/bash

shopt -s globstar

for map_file in maps/*.txt; do
    filename=$(basename "$map_file")
    number=$(echo "$filename" | sed 's/\.txt$//' | grep -E '^[0-9]+$')

    # if [[ -z "$number" ]] || [[ "$number" -lt 109 ]]; then
    #     echo "Skipping map: $map_file (number: $number)"
    #     continue
    # fi
    
    echo "Running on map: $map_file"
    python3 main.py \
        --map "$map_file" \
        --max_time 1200 \
        --num_iterations 100 \
        --algorithm unique_vis \
        --no_save \
        --no_plot \
        --save_to_db \
        --random_speeds \
        --random_budget
done

# Populate TEST database
# for map_file in benchmarks/instances/*.txt; do
#     echo "Running on map: $map_file"
#     python3 main.py \
#         --map "$map_file" \
#         --total_time 600 \
#         --num_iterations 100 \
#         --algorithm unique_vis \
#         --no_save \
#         --no_plot \
#         --out out/ \
#         --random_speeds \
#         --random_budget
# done

# python3 main.py \
#     --map "maps/1.txt" \
#     --total_time 600 \
#     --num_iterations 300 \
#     --algorithm unique_vis \
#     --out out/