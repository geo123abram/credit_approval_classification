import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from credit_approval_model import __version__ as _version
from credit_approval_model.config.core import config_values
from credit_approval_model.pipeline import make_credit_pipe
from credit_approval_model.processing.data_manager import load_pipeline
from credit_approval_model.processing.data_manager import pre_pipeline_preparation
from credit_approval_model.processing.validation import validate_inputs





def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    
    pipeline_file_name = f"{config_values.app_config_.pipeline_save_file}{_version}.pkl"
    print(_version, pipeline_file_name)
    credit_approval_pipe= load_pipeline(file_name=pipeline_file_name)
     
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))

    # DO NOT reindex here
    # validated_data = validated_data.reindex(columns=config_values.model_config_.features)

    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = credit_approval_pipe.predict(validated_data)
        results["predictions"] = predictions

    return results

if __name__ == "__main__":
    data_in={'A1':['b'],'A2':[23.25],'A3':[1.0],'A4':['u'],'A5':['g'],'A6':['c'],'A7':['v'],'A8':[0.835],'A9':['t'],'A10':['f'],'A11':[0],'A12':['f'],'A13':['s'],'A14':[300],'A15':[0]}

    print(make_prediction(input_data=data_in))
