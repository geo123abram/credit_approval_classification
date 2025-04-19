# Path setup, and access the config.yml file, datasets folder & trained models
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import credit_approval_model


# Project Directories
PACKAGE_ROOT = Path(credit_approval_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "config/datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    training_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    unused_fields: List[str]
    numerical_features: List[str]
    categorical_features: List[str]
    
    
    # declared variables
    a1_var: str
    a2_var: str
    a3_var: str
    a4_var: str
    a5_var: str
    a6_var: str
    a7_var: str
    a8_var: str
    a9_var: str
    a10_var: str
    a11_var: str
    a12_var: str
    a13_var: str
    a14_var: str
    a15_var: str

    # mappings for categorical variables
    a1_mappings: Dict[str, int]
    a4_mappings: Dict[str, int]
    a5_mappings: Dict[str, int]
    a6_mappings: Dict[str, int]   
    a7_mappings: Dict[str, int]   
    a9_mappings: Dict[str, int]   
    a10_mappings: Dict[str, int]   
    a12_mappings: Dict[str, int]   
    a13_mappings: Dict[str, int]   
    
    test_size:float
    random_state: int
    n_estimators: int
    max_depth: int
    max_features: int
    
class Config(BaseModel):
    """Master config object."""

    app_config_: AppConfig
    model_config_: ModelConfig
    
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")



def create_and_validate_config(parsed_config: YAML = None) -> Config:
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    raw: dict = parsed_config.data  # this is your flat dict of all keys

    # 1) pull out just the keys AppConfig cares about
    # app_keys = set(AppConfig.__fields__.keys())
    app_keys=set(AppConfig.model_fields.keys())
    app_dict = {k: raw[k] for k in app_keys}

    # 2) pull out just the keys ModelConfig cares about
    # model_keys = set(ModelConfig.__fields__.keys())
    model_keys = set(ModelConfig.model_fields.keys()) 
    model_dict = {k: raw[k] for k in model_keys}

    # 3) now instantiate
    return Config(
        app_config_=   AppConfig(**app_dict),
        model_config_= ModelConfig(**model_dict),
    )


config_values = create_and_validate_config()

