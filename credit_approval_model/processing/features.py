from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from credit_approval_model.config.core import config_values


class numericImputer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, variables):
        # allow either a list of feature‑names or a single feature name
        if isinstance(variables, str):
            variables = [variables]
        # now _require_ a list or tuple of str
        if not (
            isinstance(variables, (list, tuple))
            and all(isinstance(v, str) for v in variables)
        ):
            raise ValueError("variables must be a string or list of strings")
        # store the *exact* object* passed in
        self.variables = variables

    def fit(self, X, y=None):
        self.impute_values_ = {col: X[col].mean() for col in self.variables}
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.impute_values_.items():
            X[col] = X[col].fillna(val)
        return X


class categoryImputer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, variables):
        if isinstance(variables, str):
            variables = [variables]
        if not (
            isinstance(variables, (list, tuple))
            and all(isinstance(v, str) for v in variables)
        ):
            raise ValueError("variables must be a string or list of strings")
        self.variables = variables

    def fit(self, X, y=None):
        self.fill_values_ = {}
        for col in self.variables:
            try:
                mode_series = X[col].mode()
                if not mode_series.empty:
                    self.fill_values_[col] = mode_series[0]
                else:
                    self.fill_values_[col] = 'missing'  # Or any other fallback category
            except Exception as e:
                print(f"Warning: Could not compute mode for column {col}: {e}")
                self.fill_values_[col] = 'missing'
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values_.items():
            X[col] = X[col].astype(object).fillna(val)
        return X

    

class MultiMapper(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, mappings: dict[str, dict]):
        self.mappings = mappings  # ✅ store constructor param for get_params() and clone()

    def fit(self, X, y=None):
        # ✅ set the fitted attribute
        self.mappings_ = self.mappings
        return self

    def transform(self, X):
        X = X.copy()
        for col, mp in self.mappings_.items():
            X[col] = X[col].astype(object).map(mp)
            X[col] = X[col].fillna(-1).astype(int)
        return X


    
class Mapper(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):
        print(f"{variables} is type: {type(variables)}")
        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings_ = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings_).astype(int)

        return X


class A2_A3_Transformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """Creates the 'A2_A3' feature by dividing 'A2' by 'A3', handling division by zero."""

    def __init__(self, a2_col='A2', a3_col='A3', fill_value=0.0):
        if not isinstance(a2_col, str):
            raise ValueError('a2_col should be a string')
        if not isinstance(a3_col, str):
            raise ValueError('a3_col should be a string')
        if not isinstance(fill_value, (int, float)):
            raise ValueError('fill_value should be numeric')
        self.a2_col = a2_col
        self.a3_col = a3_col
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Convert '?' to np.nan before processing
        X[self.a2_col] = X[self.a2_col].replace('?', np.nan)
        X[self.a3_col] = X[self.a3_col].replace('?', np.nan)

        X[self.a2_col] = X[self.a2_col].fillna(0).astype(float)
        X[self.a3_col] = X[self.a3_col].fillna(0).astype(float)

        a2 = X[self.a2_col].to_numpy()
        a3 = X[self.a3_col].to_numpy()

        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(
                a2, a3,
                out=np.full_like(a2, self.fill_value, dtype=float),
                where=(a3 != 0)
            )

        X['A2_A3'] = result
        return X
    

class FeatureGenerator(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, a2_col='A2', a3_col='A3',
                 a8_col='A8', a11_col='A11',
                 a14_col='A14', a15_col='A15',
                 fill_value=0.0, feature_names=None):
        self.a2_col = a2_col
        self.a3_col = a3_col
        self.a8_col = a8_col
        self.a11_col = a11_col
        self.a14_col = a14_col
        self.a15_col = a15_col
        self.fill_value = fill_value
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.feature_names is None:
                raise ValueError("Feature names must be provided when input is ndarray.")
            X = pd.DataFrame(X, columns=self.feature_names)

        X = X.copy()

        # A2_A3: safe divide
        a2 = X[self.a2_col].fillna(0).astype(float).to_numpy()
        a3 = X[self.a3_col].fillna(0).astype(float).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            X['A2_A3'] = np.divide(
                a2, a3,
                out=np.full_like(a2, self.fill_value, dtype=float),
                where=(a3 != 0)
            )

        # A8_A11: multiplication
        X['A8_A11'] = X[self.a8_col].fillna(0).astype(float) * X[self.a11_col].fillna(0).astype(float)

        # A14_A15: log of product
        prod = X[self.a14_col].fillna(0).astype(float) * X[self.a15_col].fillna(0).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            logprod = np.log(prod)
        X['A14_A15'] = np.where(np.isfinite(logprod), logprod, self.fill_value)

        return X


    
class FeatureSelector(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self, features_to_keep):
        reqd_features= config_values.model_config_.features
        self.features_to_keep = reqd_features
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.features_to_keep + ['_junk'])  # dummy fallback
        return X[self.features_to_keep]