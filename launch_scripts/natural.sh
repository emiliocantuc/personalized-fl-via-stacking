#!/bin/bash

n_seeds=10

for dataset in "census" "covertype" "vehicle-loan-default"; do
        for random_state in $(seq 1 $n_seeds); do
                python3 -u main.py --data $dataset --partitioning_method natural --output_dir "results/natural/" --random_state $random_state
        done
done

echo "Done!"