from sklearn.base import BaseEstimator, TransformerMixin
# Import NumPy for the log1p transformation.
import numpy as np
import pandas as pd


# Define a reusable transformer for applying log1p to selected columns.
class LogTransformer(BaseEstimator, TransformerMixin):

    # Store the columns that will receive the log transformation.
    def __init__(self, columns):
        self.columns = columns

    # This transformer does not learn parameters from the training data.
    def fit(self, X, y=None):
        return self

    # Apply log1p to each configured column and return a copied DataFrame.
    def transform(self, X):
        X = X.copy()

        for column in self.columns:
            X[column] = np.log1p(X[column])

        return X
    
    
# Define a reusable transformer that converts invalid zero values to NaN.
class ZeroToNaNTransformer(BaseEstimator, TransformerMixin):

    # Store the columns where zero represents a missing measurement.
    def __init__(self, columns):
        self.columns = columns

    # This transformer does not learn parameters from the training data.
    def fit(self, X, y=None):
        return self

    # Replace zero values in the configured columns and return a copy.
    def transform(self, X):
        X = X.copy()

        for column in self.columns:
            X[column] = X[column].replace(0, np.nan)

        return X
    


# Define a wrapper that preserves column names and row indexes after transformation.
class DataFrameTransformer(BaseEstimator, TransformerMixin):

    # Store the sklearn transformer and the expected output columns.
    def __init__(self, transformer, columns):
        self.transformer = transformer
        self.columns = columns

    # Fit the wrapped transformer on the input data.
    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    # Transform the data and restore its DataFrame structure.
    def transform(self, X):
        X_transformed = self.transformer.transform(X)

        return pd.DataFrame(
            X_transformed,
            columns=self.columns,
            index=X.index
        )