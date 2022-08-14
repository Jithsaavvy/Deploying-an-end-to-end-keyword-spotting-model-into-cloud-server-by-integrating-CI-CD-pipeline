#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Defines and create model for training and evaluation.
`CNN-LSTM` is used for this project with 1D convolutional layers
followed by LSTM layers with self-attention and fully connected
layers. This script provides the flexibility to add any other 
models by inheriting Model(ABC). 
"""

from dataclasses import dataclass
from typing import Tuple
from tensorflow import keras
from abc import ABC, abstractmethod
from keras.models import Model, Sequential
from keras_self_attention import SeqSelfAttention
from keras.layers import  Conv1D, MaxPooling1D, LSTM
from keras.layers import Input, Dropout, BatchNormalization, Dense

class Model(ABC):
    """
    Abstract base class that defines and creates model.
    """
    @abstractmethod
    def define_model(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

@dataclass
class CNN_LSTM_Model(Model):
    """
    Dataclass to create CNN-LSTM model that inherits Model class.
    """
    input_shape: Tuple[int, int]
    num_classes: int

    def define_model(self) -> Sequential:
        """
        Method to define model that can be used for training
        and inference. This existing model can also be tweaked 
        by changing parameters, based on the requirements.

        Parameters
        ----------
            None.

        Returns
        -------
        Sequential
        """

        return Sequential(
            [
            Input(shape=self.input_shape),
            BatchNormalization(),

            #1D Convolutional layers
            Conv1D(32, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3),
            Conv1D(64, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3),
            Conv1D(128, kernel_size=3, strides=1, padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size = 3, padding='same'),
            Dropout(0.30),
            
            #LSTM layers
            LSTM(units = 128, return_sequences=True),
            SeqSelfAttention(attention_activation='tanh'),
            LSTM(units = 128, return_sequences=False),
            BatchNormalization(),
            Dropout(0.30),

            #Dense layers
            Dense(256, activation='relu'),
            Dense(64, activation='relu'),
            Dropout(0.30),
            Dense(self.num_classes, activation='softmax')
            ]
        )

    def create_model(self) -> Sequential:
        """
        Method to create the model defined by define_model() method
        and prints the model summary.

        Parameters
        ----------
            None.

        Returns
        -------
        model: Sequential
        """
        model: Sequential = self.define_model()
        model.summary()
        return model