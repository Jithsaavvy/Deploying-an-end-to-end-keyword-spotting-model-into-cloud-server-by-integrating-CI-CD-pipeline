"""
@author: Jithin Sasikumar

File to define type hints for all configurations in Hydra.
It is an optional file.
"""

from dataclasses import dataclass

@dataclass
class Paths:
    train_dir: str
    test_dir: str
    mlflow_tracking_uri: str
    model_artifactory_dir: str
    audio_dir: str

@dataclass
class Params:
    epochs: int
    learning_rate: float
    test_data_split_percent: float
    mfcc_length: int
    sampling_rate: int
    n_mfcc: int
    batch_size: int

@dataclass
class Names:
    experiment_name: str
    audio_file: str
    metric_name: str

@dataclass
class KWSConfig:
    paths: Paths
    params: Params
    names: Names