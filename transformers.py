# Custom sklearn transformers
from sklearn.base import BaseEstimator, TransformerMixin


class MismatchedColumnsHandler(BaseEstimator, TransformerMixin):
    """
    Add missing columns with default values and remove extra columns.
    Note: Almost a SimpleImputer but with the ability to remove extra columns.
    """
    def __init__(self, default_values):
        """
        Will add missing columns with default values and remove extra columns
        Args:
            input_cols: list / array of columns in df to be inputed into model
            default_values: dictionary with default values for columns the model expects
        """
        self.default_values = default_values

    def fit(self, X = None, y = None):
        return self

    def transform(self, X):

        if set(self.default_values.keys()) == set(X.columns):
            return X
        
        X = X.copy() # Avoid changing the original dataframe

        # Missing columns: Expected by the model but not in input
        missing_cols = list(set(self.default_values.keys()) - set(X.columns))
        
        # Add missing columns with default values
        for col in missing_cols:
            X[col] = self.default_values[col]
        
        # Just keep the columns that the model expects
        X = X[self.default_values.keys()]

        return X
    

if __name__ == '__main__':

    import pandas as pd
    import numpy as np
    
    # Test MismatchedColumnsHandler
    t = MismatchedColumnsHandler(default_values={'2': -1, '3': -1, '4': 4})
    X = pd.DataFrame({'1': [1], '2': [2], '3': [3]})
    expected = pd.DataFrame({'2': [2], '3': [3], '4': [4]})
    print(t.transform(X))
    assert t.transform(X).equals(expected), 'MismatchedColumnsHandler failed'