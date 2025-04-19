import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import re
import joblib
import pandas as pd
import typing as t
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
import numpy as np


from credit_approval_model import __version__ as _version
from credit_approval_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config_values


##  Pre-Pipeline Preparation

# do one-hot encoding for categorical variables
def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df= df.copy()
    # Identify categorical columns
    text_cols = config_values.model_config_.categorical_features
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Encode categorical columns
    for col in text_cols:
        df[col] = label_encoder.fit_transform(df[col])  
    return df

def pre_pipeline_preparation(*, data_frame: pd.DataFrame) -> pd.DataFrame:
    data_frame1 = data_frame.copy()
    data_frame1=data_frame1.replace("?", pd.NA)  # Replacing "?" with NaN
    
    numeric_cols=config_values.model_config_.numerical_features
    # print(f"Numeric columns: {numeric_cols}")
    # Convert numeric columns to float
    for col in numeric_cols:
        data_frame1[col] = pd.to_numeric(data_frame1[col], errors='coerce')
        
    # Convert categorical columns to category type
    categorical_cols = config_values.model_config_.categorical_features
    # print(f"Categorical columns: {categorical_cols}")
    for col in categorical_cols:
        data_frame1[col] = data_frame1[col].astype('category')
    
    data_frame1['A2_A3'] =0.0 
    data_frame1['A8_A11'] =0.0 
    data_frame1['A14_A15'] =0.0 
    
    return data_frame1



def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe.columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'credit_category']
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe.columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'credit_category']
    transformed = pre_pipeline_preparation(data_frame=dataframe)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config_values.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


# def load_pipeline(*, file_name: str) -> Pipeline:
#     """Load a persisted pipeline."""

#     file_path = TRAINED_MODEL_DIR / file_name
#     print(f"Loading model from {file_path}")
#     trained_model = joblib.load(filename=file_path)
#     return trained_model
def load_pipeline(file_name: str):
    file_path = TRAINED_MODEL_DIR / file_name
    # print(f"[DEBUG] Attempting to load pipeline from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    pipeline = joblib.load(file_path)
    # print(f"[DEBUG] Pipeline loaded: {type(pipeline)}")

    # Check if the loaded pipeline is actually fitted
    try:
        check_is_fitted(pipeline)
        # print("[DEBUG] Pipeline is fitted")
    except Exception as e:
        print(f"[DEBUG] Pipeline is NOT fitted: {e}")

    return pipeline


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

