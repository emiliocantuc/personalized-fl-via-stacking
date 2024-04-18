# Personalized Federated Learning via Stacking

This repository contains the code for *(Personalized Federated Learning via Stacking)[https://arxiv.org/abs/2404.10957)*.

## Structure

- `main.py`: Main script for launching a single simulated federation, typically used with scripts in `launch_scripts/`.
- `data.py`: Handles dataset fetching and preprocessing.
- `partitioning.py`: Contains functions for natural, quantity skew, label skew, and vertical partitioning.
- `transformers.py`: Includes a custom scikit-learn transformer for dealing with vertically partitioned data.
- `figs.ipynb`: Processes results and generates figures as presented in the paper.

## Reproducibility

To reproduce the results and figures from the paper, follow these steps:

1. Install Python package requirements with `pip install -r requirements.txt`.
2. Obtain a Kaggle API key by following [these instructions](https://www.kaggle.com/docs/api).
3. Run `sh launch_scripts/all.sh`.
4. Open and execute `figs.ipynb`.
