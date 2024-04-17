#!/bin/bash

n_seeds=10

for dataset in "census" "covertype" "vehicle-loan-default"; do
        for alpha in 0.5 1.0 10.0 100.0; do
                for random_state in $(seq 1 $n_seeds); do
                        python3 -u main.py --data $dataset --partitioning_method dirichlet --partitioning_params "{\"alpha\":$alpha, \"c_clients\": 10}" --output_dir "results/dirichlet/" --random_state $random_state
                done
        done
done

echo "Done!"