#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to perform model training.
"""

from tensorflow import keras
from keras import optimizers
from src.model import CNN_LSTM_Model
from src.data import Dataset
from src.exception_handler import ValueError
from src.experiment_tracking import MLFlowTracker, ModelSelection

class Training:
    def __init__(self, model: CNN_LSTM_Model, dataset: Dataset,
                batch_size: int, epochs: int, learning_rate: float,
                tracker: MLFlowTracker, metric_name: str) -> None:
        """
        Instance variables
        ------------------
        model: CNN_LSTM_Model
            Instance of CNN_LSTM_Model class holding the created model.
        dataset: Dataset
            Instance of Dataset class holding the processed data(train & test).
        batch_size: int
            Number of samples per gradient update.
        epochs: int
            Number of epochs to train the model.
        learning_rate: float
            Rate of model training.
        tracker: MLFlowTracker
            Instance of MLFlowTracker class.
        metric_name: str
            Metric name to sort the models.

        Returns
        -------
            None.
        """
        self.model = model
        self.dataset_ = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tracker = tracker
        self.metric_name = metric_name
        
    def train(self) -> ModelSelection:
        """
        Method that initializes and performs training.

        Parameters
        ----------
            None.
        
        Returns
        -------
        instanceof(ModelSelection):
            Instance will hold resulting best model information after selecting from the
            model artifacts based on the given metric.

        Raises
        ------
        ValueError: Exception
            If self.metric_name is not given or null.
        """

        if self.metric_name is None:
            raise ValueError("Please provide the metric name for model selection !!!")
            
        print("Training started.....")
        self.model.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.Nadam(learning_rate=self.learning_rate),
                        metrics=['accuracy'])
                        
        history = self.model.fit(self.dataset_.x_train, self.dataset_.y_train,
                        batch_size = self.batch_size,
                        epochs = self.epochs,
                        verbose = 1,
                        validation_data = (self.dataset_.x_test, self.dataset_.y_test))

        return ModelSelection(self.tracker.find_best_model(self.metric_name))