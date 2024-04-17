#!/bin/bash

n_seeds=10

for dataset in "census" "covertype" "vehicle-loan-default"; do
    for a in 0.25 0.5 0.75 1.0; do
        for random_state in $(seq 1 $n_seeds); do
            python3 -u main.py --data $dataset --partitioning_method power --partitioning_params "{\"a\":$a, \"c_clients\": 10}" --output_dir "results/power/" --random_state $random_state
        done
    done
done

echo "Done!"