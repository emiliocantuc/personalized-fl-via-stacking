#!/bin/bash

n_seeds=10
for dataset in "covertype" "census" "vehicle-loan-default"; do
        for random_state in $(seq 1 $n_seeds); do
                for prop_cols in 0.25 0.5 0.75; do
                        for eps in 0.0 2.0 4.0 8.0 16.0; do                      
                                time python3 -u main.py --data $dataset --partitioning_method vertical --eps $eps --partitioning_params "{\"prop_cols\":$prop_cols, \"c_clients\": 10}" --output_dir "results/vertical/" --random_state $random_state
                        done
                done
        done
done

echo "Done!"