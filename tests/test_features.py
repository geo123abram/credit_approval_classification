
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import pandas as pd
from credit_approval_model.config.core import config_values as config
from credit_approval_model.processing.features import A2_A3_Transformer


def test_A2_A3_Transformer(sample_input_data):
    X_test, _ = sample_input_data

    # Dynamically select a row with missing values in A2 or A3
    row_idx = X_test[
        (X_test[config.model_config_.a2_var].isin(['?', None])) |
        (X_test[config.model_config_.a3_var].isin(['?', None]))
    ].index[0]

    transformer = A2_A3_Transformer(
        a2_col=config.model_config_.a2_var,
        a3_col=config.model_config_.a3_var
    )

    transformed = transformer.fit(X_test).transform(X_test)

    assert 'A2_A3' in transformed.columns
    assert isinstance(transformed.loc[row_idx, 'A2_A3'], float)
