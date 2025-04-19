from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Adjust Python path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Custom modules
from credit_approval_model.config.core import config_values
from credit_approval_model.processing.features import (
    numericImputer,
    categoryImputer,
    MultiMapper,
    FeatureGenerator,
    FeatureSelector,
)


def get_ct_feature_names(column_transformer):
    """
    Extracts feature names from a fitted ColumnTransformer.
    Handles nested pipelines (e.g., for categorical preprocessing).
    """
    output_features = []

    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop" or transformer is None:
            continue

        if isinstance(transformer, SKPipeline):
            last_step = transformer.steps[-1][1]
            try:
                check_is_fitted(last_step)
                if hasattr(last_step, "get_feature_names_out"):
                    features = last_step.get_feature_names_out(columns)
                else:
                    features = columns
            except NotFittedError:
                features = columns
        else:
            try:
                check_is_fitted(transformer)
                if hasattr(transformer, "get_feature_names_out"):
                    features = transformer.get_feature_names_out(columns)
                else:
                    features = columns
            except NotFittedError:
                features = columns

        output_features.extend(features if isinstance(features, (list, np.ndarray)) else [features])

    return output_features


def make_credit_pipe():
    cm = config_values.model_config_
    num_vars = cm.numerical_features
    cat_vars = cm.categorical_features

    # Category pipeline (impute + map)
    mappings = {
        var: getattr(cm, f"{var.lower()}_mappings") for var in cat_vars
    }
    cat_pipeline = SKPipeline([
        ("impute", categoryImputer(variables=cat_vars)),
        ("map", MultiMapper(mappings=mappings)),
    ])

    # ColumnTransformer
    preproc = ColumnTransformer([
        ("num_imp", numericImputer(variables=num_vars), num_vars),
        ("cat_proc", cat_pipeline, cat_vars),
    ], remainder="drop")

    # Dummy data to trigger internal shape/fit state
    dummy_data = pd.DataFrame({col: [np.nan] for col in num_vars + cat_vars})
    dummy_pipe = SKPipeline([
        ("preproc", preproc)
    ])
    dummy_pipe.fit(dummy_data)

    feature_names = get_ct_feature_names(dummy_pipe.named_steps["preproc"])

    # Final model pipeline
    return SKPipeline([
        ("preproc", dummy_pipe.named_steps["preproc"]),  # use fitted ColumnTransformer
        ("feature_gen", FeatureGenerator(
            a2_col="A2", a3_col="A3",
            a8_col="A8", a11_col="A11",
            a14_col="A14", a15_col="A15",
            feature_names=feature_names
        )),
        ("select_features", FeatureSelector(
            features_to_keep=cm.features
        )),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=cm.n_estimators,
            max_depth=cm.max_depth,
            max_features=cm.max_features,
            random_state=cm.random_state
        )),
    ])
