import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split

from credit_approval_model.config.core import config_values
from credit_approval_model.processing.data_manager import load_raw_dataset


# @pytest.fixture is a decorator in the pytest testing framework used to define a fixture. 
# Fixtures are a way to provide setup and teardown functionality for your test functions or test classes. 
# They allow you to create reusable components that can be shared across multiple test functions, 
# improving code modularity and maintainability.

# @pytest.fixture
# def sample_input_data():
#     dataset_path = Path(__file__).resolve().parent.parent / "credit_approval_model" / "config" / "datasets" / config_values.app_config_.training_data_file
#     data = load_raw_dataset(file_name= dataset_path)

#     X = data.drop(config_values.model_config_.target, axis=1)       # predictors
#     y = data[config_values.model_config_.target]                    # target

#     # divide train and test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,  # predictors
#         y,  # target
#         test_size=config_values.model_config_.test_size,
#         # we are setting the random seed here
#         # for reproducibility
#         random_state=config_values.model_config_.random_state,
#     )

#     return X_test, y_test

@pytest.fixture
def sample_input_data():
    dataset_path = config_values.app_config_.training_data_file  # just the filename
    data = load_raw_dataset(file_name=dataset_path)

    X = data.drop(config_values.model_config_.target, axis=1)
    y = data[config_values.model_config_.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config_values.model_config_.test_size,
        random_state=config_values.model_config_.random_state,
    )

    return X_test, y_test
