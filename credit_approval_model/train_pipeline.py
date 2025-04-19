import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from credit_approval_model.config.core import config_values
from credit_approval_model.pipeline import make_credit_pipe
from credit_approval_model.processing.data_manager import load_dataset, save_pipeline

def run_training(data_file_name=config_values.app_config_.training_data_file) -> None:
    
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=data_file_name)
    # print(data.info(show_counts=True))
    
        # split the raw X and y, not only the final features
    X = data.drop(columns=[config_values.model_config_.target])
    y = data[config_values.model_config_.target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = config_values.model_config_.test_size,
        random_state = config_values.model_config_.random_state
    )
 
    # # Pipeline fitting
    credit_approval_pipe = make_credit_pipe()
    credit_approval_pipe.fit(X_train,y_train)
    y_pred = credit_approval_pipe.predict(X_test)
    result=("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # # persist trained model
    save_pipeline(pipeline_to_persist= credit_approval_pipe)
    # # printing the score
    return result
    
if __name__ == "__main__":
    run_training()
