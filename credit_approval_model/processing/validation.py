import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from credit_approval_model.config.core import config_values
from credit_approval_model.processing.data_manager import pre_pipeline_preparation




def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    pre_processed = pre_pipeline_preparation(data_frame=input_df)
    
    # Just keep all columns as-is
    validated_data = pre_processed.copy()
    
    errors = None
    try:
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInputSchema(BaseModel):
    # 'A15', 'A14', 'A8', 'A2', 'A11', 'A3'
    A1: Optional[str]
    A2: Optional[float]
    A3: Optional[float]
    # A2_A3: Optional[float]
    A4: Optional[str]
    A5: Optional[str]
    A6: Optional[str]
    A7: Optional[str]
    A8: Optional[float]
    # A8_A11: Optional[float]
    A9: Optional[str]
    A10: Optional[str]
    A11: Optional[int]
    A12: Optional[str]
    A13: Optional[str]
    A14: Optional[float]
    A15: Optional[float]
    # A14_A15: Optional[float]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
