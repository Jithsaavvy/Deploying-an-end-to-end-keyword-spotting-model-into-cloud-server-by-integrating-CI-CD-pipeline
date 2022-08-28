#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to load audio dataset, preprocess them and train CNN-LSTM
model and log the metrics. The resulting metrics, parameters
and model artifacts are tracked and logged via MLFlow.

Note: Change the parameters and paths accordingly in
`./config_dir/config.yaml` based on the requirements.
"""

import warnings
import hydra
from keras.models import Sequential
from hydra.core.config_store import ConfigStore
from config_dir.configType import KWSConfig
from src import train
from src import data
from src.model import CNN_LSTM_Model
from src.experiment_tracking import MLFlowTracker, ModelSelection
warnings.filterwarnings('ignore')

cs = ConfigStore.instance()
cs.store(name="kws_config", node=KWSConfig)

@hydra.main(config_path="config_dir", config_name="config")
def main(cfg: KWSConfig) -> None:
    """
    Function to initialize the training pipeline.
 
    The dataset used for training will be the ones dumped as
    `.npy` files from original audio files for training in
    `./dataset/train`. More explanation on this is provided
    in `README.md` and refer code in `./src/data.py`.

    Parameters
    ----------
    cfg: KWSConfig
        Instance of KWSConfig.
        `Hydra` framework is used for configuration management
        across the application. It facilitates to create a
        hierarchical configurations and store them which is easier
        and handy to access. It's a good practice to use such
        configuration management tools which helps to nullify the
        hassles and perplexity, caused by too many configurations
        for a single application. For more information, visit
        @https://hydra.cc/

    Returns
    -------
        None

    Raises
    ------
    exc: Exception
        Proper exception handling is done in every respective files.
    """
    try:
        #Initializing MLFlow for model tracking and logging
        tracker = MLFlowTracker(cfg.names.experiment_name, cfg.paths.mlflow_tracking_uri)
        tracker.log()
       
        #Load and preprocess the audio dataset for training
        dataset_ = data.Dataset()
        preprocess_ = data.Preprocess(dataset_,cfg.paths.train_dir, cfg.params.n_mfcc,
                                cfg.params.mfcc_length, cfg.params.sampling_rate)                   
        preprocessed_dataset: data.Dataset = preprocess_.preprocess_dataset(preprocess_.labels,
                                                    cfg.params.test_data_split_percent)
        [data.print_shape(key, value) for key, value in preprocessed_dataset.__dict__.items()]

        #Loading and training the model
        model: Sequential = CNN_LSTM_Model((cfg.params.n_mfcc, cfg.params.mfcc_length),
                                        len(preprocess_.labels)).create_model()
        best_selected_model: ModelSelection  = train.Training(model, preprocessed_dataset,
                                                             cfg.params.batch_size,
                                                             cfg.params.epochs,
                                                             cfg.params.learning_rate,
                                                             tracker,
                                                             cfg.names.metric_name).train()

    except Exception as exc:
        raise Exception("ffhffh") from exc
                         
if __name__ == "__main__":
    main()