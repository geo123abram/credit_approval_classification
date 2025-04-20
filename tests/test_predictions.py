"""
Note: These tests will fail if you have not first trained the model.
"""

import sys
from pathlib import Path
from credit_approval_model.predict import make_prediction
import numpy as np
import pandas as pd
# from sklearn.metrics import accuracy_score  # Uncomment if you're using y_true later

# Setup path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))


def test_make_prediction(sample_input_data=None):

    # Given
    if sample_input_data is None:
        sample_input_data = {
            'A1': ['b'], 'A2': [23.25], 'A3': [1.0], 'A4': ['u'],
            'A5': ['g'], 'A6': ['c'], 'A7': ['v'], 'A8': [0.835],
            'A9': ['t'], 'A10': ['f'], 'A11': [0], 'A12': ['f'],
            'A13': ['s'], 'A14': [300], 'A15': [0]
        }

    input_df = pd.DataFrame(sample_input_data)

    # When
    result = make_prediction(input_data=input_df)

    # Then
    predictions = result.get("predictions")

    assert type(predictions) is list
    assert isinstance(predictions[0], str)
    assert predictions[0] in ['+', '-']
    assert result.get("errors") is None
    assert len(predictions) == len(input_df)

    # Optional: Uncomment if you have actual labels to compare
    # y_true = [...]
    # accuracy = accuracy_score(predictions, y_true)
    # assert accuracy > 0.8
