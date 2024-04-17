import argparse, json, os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

import data, partitioning, transformers


EVAL_METRICS = {
    'ROC AUC': roc_auc_score,
    'Balanced Accuracy': balanced_accuracy_score,
    'Accuracy': accuracy_score
}

def score(m, X, y):
    return {name: metric(y, m.predict(X)) for name, metric in EVAL_METRICS.items()}


def train_val_test_split(y, val_size, test_size, random_state, stratify = False):
    """Split the data into train, val, and test sets"""
    
    train, test = train_test_split(
        y, test_size = test_size, random_state = random_state,
        stratify = y if stratify else None
    )

    train, val = train_test_split(
        train, test_size = val_size / (1 - test_size), random_state = random_state,
        stratify = train if stratify else None
    )
    
    return train.index, val.index, test.index

def default_feature_values(X, eps, random_state):

    cat_cols = X.dtypes[(X.dtypes == 'object') | (X.dtypes == 'bool')].index.tolist()
    num_cols = X.dtypes[(X.dtypes != 'object') & (X.dtypes != 'bool')].index.tolist()
    default_vals = X.iloc[0].copy()
    np.random.seed(random_state)
    
    # Default value for numerical columns is the median + N(0, eps)
    if num_cols:
        default_vals[num_cols] = X[num_cols].median() + np.random.normal(0, X[num_cols].std() * eps, X[num_cols].shape[1])

    # And for categorical columns, it's the mode
    if cat_cols:
        modes = X[cat_cols].mode()
        for col in cat_cols:
            default_vals[col] = modes[col].values[0]

    return default_vals


class Island:

    def __init__(self, name):
        self.name = name
        self.public_model = None
        self.default_col_values = {}
        self.canonical_cols = []


def main(data_obj, df, partition, val_size, test_size, n_bootstraps, eps, random_state):

    has_natural_col = False
    if hasattr(data_obj, 'natural_col'):
        has_natural_col = True
        partition = [(ix, [c for c in cols if c != data_obj.natural_col]) for ix, cols in partition]

    islands = []
    results = []

    # Assign each of them their own pipeline
    for i, (ix, cols) in enumerate(partition):
        
        island = Island(i)
        X = df.loc[ix, cols].drop(columns = [data_obj.target])
        y = df.loc[ix, data_obj.target]

        island.canonical_cols = partitioning._canonical_cols(X, skip_cols = [data_obj.natural_col] if has_natural_col else [])

        # Each island fits its public model on ALL its data
        island.public_model = RandomForestClassifier(random_state = random_state, n_jobs = 1).fit(X, y) # Could use pipeline here to preprocess
        island.default_col_values = default_feature_values(X, eps, random_state)
        assert island.default_col_values.index.equals(X.columns)

        islands.append(island)

    # Each island
    for i, (island, (ix, cols)) in enumerate(zip(islands, partition)):

        print(f'Island {i + 1} of {len(islands)}')

        X = df.loc[ix, cols].drop(columns = [data_obj.target])
        y = df.loc[ix, data_obj.target]

        # Island's private model
        private_m = RandomForestClassifier(random_state = random_state, n_jobs = -1)

        # 'baseline': [{'accuracy': 0.8, 'balanced_accuracy': 0.8}, ...]
        cv_scores = {i:[] for i in ['baseline', 'stack_on_validation', 'stack_on_pooled']}
        importances_on_val, importances_on_pooled = np.zeros(len(islands)), np.zeros(len(islands))

        for rs in range(n_bootstraps):

            train, validation, test = train_val_test_split(
                y, val_size, test_size, random_state = (random_state * n_bootstraps) + rs,
                stratify = data_obj.is_classification
            )

            # For baseline train private model on train + validation and score on test
            pooled_train_validation = train.append(validation)
            private_m.fit(X.loc[pooled_train_validation], y.loc[pooled_train_validation])
            cv_scores['baseline'].append(score(private_m, X.loc[test], y.loc[test]))
            
            # Base estimators to stack (others are already fitted whereas private_m is not)
            base_estimators = [(str(other.name),
                Pipeline(
                    [
                        # We can "fit" the mismatched columns handler with None
                        ('match-cols', transformers.MismatchedColumnsHandler(other.default_col_values).fit()),
                        # ('preprocessing',...) # rescale numerical features?
                        ('model', other.public_model)
                    ]
                )) if j != i else (str(other.name), private_m)
                for j, other in enumerate(islands)
            ]

            # Stack on VALIDATION SET
            base_estimators[i][1].fit(X.loc[train], y.loc[train]) # Train private on training set
            meta = StackingClassifier(
                estimators = base_estimators,
                final_estimator = RandomForestClassifier(random_state = random_state, n_jobs = -1), cv = 'prefit'
            )
            meta.fit(X.loc[validation], y.loc[validation]) # Stack on the validation set
            cv_scores['stack_on_validation'].append(score(meta, X.loc[test], y.loc[test]))
            importances_on_val += meta.final_estimator_.feature_importances_

            # Stack on POOLED (train + validation) SET
            base_estimators[i][1].fit(X.loc[pooled_train_validation], y.loc[pooled_train_validation]) # Train private on pooled set
            meta = StackingClassifier(
                estimators = base_estimators,
                final_estimator = RandomForestClassifier(random_state = random_state, n_jobs = -1), cv = 'prefit'
            )
            meta.fit(X.loc[pooled_train_validation], y.loc[pooled_train_validation]) # Stack on the pooled set
            cv_scores['stack_on_pooled'].append(score(meta, X.loc[test], y.loc[test]))
            importances_on_pooled += meta.final_estimator_.feature_importances_

        
        island_result = {
            'island': i,
            'n': X.shape[0],
            'n_fraction': X.shape[0] / df.shape[0],
            'p': X.shape[1],
            'y_mean': y.mean(),
        }

        # Log metrics
        for metric in EVAL_METRICS:

            baseline = np.array([d[metric] for d in cv_scores['baseline']])
            stack_on_validation = np.array([d[metric] for d in cv_scores['stack_on_validation']])
            stack_on_pooled = np.array([d[metric] for d in cv_scores['stack_on_pooled']])

            # Original scores
            for fname, f in [('mean', np.mean), ('std', np.std)]:
                
                # Original scores
                island_result[f'baseline_{metric}_{fname}'] = f(baseline)
                island_result[f'stack_on_validation_{metric}_{fname}'] = f(stack_on_validation)
                island_result[f'stack_on_pooled_{metric}_{fname}'] = f(stack_on_pooled)

                # Deltas
                island_result[f'stack_on_pooled_delta_{metric}_{fname}'] = f(stack_on_pooled - baseline)
                island_result[f'stack_on_validation_delta_{metric}_{fname}'] = f(stack_on_validation - baseline)

        
        # Log importances
        importances_on_val /= importances_on_val.sum()
        importances_on_pooled /= importances_on_pooled.sum()
        for j, (on_val, on_pooled) in enumerate(zip(importances_on_val, importances_on_pooled)):
            island_result[f'imp_on_validation_{j}'] = on_val
            island_result[f'imp_on_pooled_{j}'] = on_pooled

        # Log jaccard similarity with other islands' columns
        for j, other in enumerate(islands):
            my_cols, other_cols = set(island.canonical_cols), set(other.canonical_cols)
            island_result[f'jaccard_{j}'] = len(my_cols.intersection(other_cols)) / len(my_cols.union(other_cols))
        
        results.append(island_result)

    return results


if __name__ == '__main__':

    datasets = {'census': data.Census, 'covertype': data.Covertype, 'vehicle-loan-default': data.Vehicle_Loan_Default}
    partitioning_methods = {
        'natural': partitioning.natural_partition,
        'power': partitioning.power_partition_n,
        'dirichlet': partitioning.dirichlet_partition,
        'vertical': partitioning.vertical_partitioning,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_state', type = int, default = 42)
    parser.add_argument('--data', type = str, choices = datasets.keys())

    parser.add_argument('--partitioning_method', type = str, choices = partitioning_methods.keys())
    parser.add_argument('--partitioning_params', type = json.loads, default = '{}')

    parser.add_argument('--val_size', type = float, default = 0.2)
    parser.add_argument('--test_size', type = float, default = 0.2)
    parser.add_argument('--n_bootstraps', type = int, default = 5)
    parser.add_argument('--eps', type = float, default = 0.0)
    parser.add_argument('--output_dir', type = str, default = 'results/tmp')
    args = parser.parse_args()


    data_obj = datasets[args.data]()
    df = data_obj.df()


    partitioning_params = args.partitioning_params
    if args.partitioning_method == 'natural':
        partitioning_params['natural_col'] = data_obj.natural_col
    elif args.partitioning_method == 'dirichlet':
        partitioning_params['y'] = df[data_obj.target]
    elif args.partitioning_method == 'vertical':
        partitioning_params['target_col'] = data_obj.target
        if hasattr(data_obj, 'natural_col'):
            partitioning_params['natural_col'] = data_obj.natural_col

    if args.partitioning_method == 'vertical' and args.data == 'covertype':
        # Choose a random subset of 50k rows for computational reasons
        # (it was super slow with 500k examples)
        df = df.sample(50000, random_state = 42) 
        df = df.reset_index(drop = True)

    partitioning_params['random_state'] = args.random_state

    # Print dir of params
    p_param_str = {k:v if k != 'y' else 'df' for k, v in partitioning_params.items()}
    print(f'Launching run with ' + repr({k:v if k != 'partitioning_params' else p_param_str for k, v in vars(args).items()}))

    partition = partitioning_methods[args.partitioning_method](df, **partitioning_params)
    results = main(data_obj, df, partition, args.val_size, args.test_size, args.n_bootstraps, args.eps, args.random_state)
    results = pd.DataFrame(results)

    # Add the parameters to the results
    for k, v in vars(args).items():
        if k != 'partitioning_params':
            results[k] = v
    for k, v in partitioning_params.items():
        results[k] = v
    results = results.drop(columns = ['output_dir'])
 
    os.makedirs(args.output_dir, exist_ok = True)
    fname = str(len(os.listdir(args.output_dir))) + '.csv'
    results.to_csv(os.path.join(args.output_dir, fname), index = False)
    print(f'Wrote to {os.path.join(args.output_dir, fname)}')
    