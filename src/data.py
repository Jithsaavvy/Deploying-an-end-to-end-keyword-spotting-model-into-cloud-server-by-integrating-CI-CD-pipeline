#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Load, preprocess audio dataset and extract features. The audio files are loaded,
each file is preprocessed, and dumped as `.npy` files which is convenient to work 
with. Thus, dumped `.npy` files can be loaded and performed with some additional 
preprocessing steps and can be used for training.
"""

import os
import librosa
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from keras.utils import to_categorical
from typing import List, Any, Tuple
from sklearn.model_selection import train_test_split
from src.exception_handler import DirectoryError, ValueError

@dataclass
class Dataset:
    """
    Dataclass that represent a dataset which is flexible to be used
    for any model training. 
    """
    x_train: np.ndarray = None
    y_train: np.array = None
    x_test: np.ndarray = None
    y_test: np.array = None

@dataclass
class Preprocess:
    """Preprocess audio dataset to be used for training.
    """
    dataset_: Dataset = None
    train_dir: str = "./dataset/train/"
    n_mfcc: int = 49
    mfcc_length: int = 40
    sampling_rate: int = 8000
    extension: str = ".npy"

    def __post_init__(self) -> None:
        """
        Dunder method to perform exception handling to catch invalid directory.

        Returns:
            None.

        Raises
        ------
        DirectoryError: Exception
            If self.train_dir does not exist.
        """
        if not os.path.exists(self.train_dir):
            raise DirectoryError(
                f"{self.train_dir} doesn't exists. Please enter a valid path !!!")

    @property
    def labels(self) -> List:
        """
        Class property to return the labels from data.

        Returns
        -------
        List of labels.
        """
        return ['.'.join(file_.split('.')[:-1]) 
               for file_ in os.listdir(self.train_dir) 
               if os.path.isfile(os.path.join(self.train_dir, file_)) 
               and check_fileType(filename = file_, extension = self.extension)]

    def __load_dataset(self, labels: List,
                       load_format: str = ".npy") -> Tuple[np.ndarray]:
        """
        Private method to load `.npy` files to preprocess.

        Parameters
        ----------
        labels: List
            List of labels.
        load_format: str
            Format to load from disk. Defaults to `.npy`.

        Returns
        -------
        data, labels: Tuple[np.ndarray]
            Tuple representing data(X) and its labels(y).
        """
        data = np.load(f"{self.train_dir + labels[0] + load_format}")
        labels = np.zeros(data.shape[0])
        for index, label in enumerate(self.labels[1:]):
            x = np.load(f"{self.train_dir + label + load_format}")
            data = np.vstack((data, x))
            labels = np.append(labels, np.full(x.shape[0], 
                          fill_value = (index + 1)))

        return data, labels
    
    def preprocess_dataset(self, labels: List, 
                            test_split_percent: float) -> Dataset:
        """
        Preprocess the loaded dataset.

        Parameters
        ----------
        labels: List
            List of labels.
        test_split_percent: float
            Train-test split percentage/ratio.

        Returns
        -------
        instanceof(Dataset):
            Instance of Dataset after preprocessing.
            The labels are one-hot encoded.

        Raises
        ------
        ValueError: Exception
            If loaded dataset is empty or null.
        """

        X, y = self.__load_dataset(labels)
        x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                           test_size = test_split_percent, 
                                           random_state=42, shuffle = True)

        for data in (x_train, x_test, y_train, y_test):
            if data is None:
                raise ValueError(f"{data} is null. Please check and preprocess again!!!")

        return Dataset(x_train, to_categorical(y_train, num_classes = len(labels)),
                       x_test, to_categorical(y_test, num_classes = len(labels)))

    
    def dump_audio_files(self, audio_files_dir: str, labels: List, n_mfcc: int, 
                        mfcc_length: int, sampling_rate: int, 
                        save_format: str = ".npy") -> None:
        """
        Method to load, process and dump audio files as `.npy` for training.
        This method is `optional` and used only with audio files. If not, skip this 
        method and use preprocess_dataset() directly for training.

        Returns
        -------
            None.
        """
        for label in labels:
            mfcc_features_np = list()
            audio_files = [audio_files_dir + label + '/' + audio_file 
                           for audio_file in os.listdir(audio_files_dir + '/' + label)]
            for audioFile in tqdm(audio_files):
                mfcc_features = convert_audio_to_mfcc(audioFile, n_mfcc, 
                                                      mfcc_length, sampling_rate)
                mfcc_features_np.append(mfcc_features)
            np.save(f"{self.train_dir + label + save_format}", mfcc_features_np)

        print(f".npy files dumped to {self.train_dir}")

    def wrap_labels(self) -> List:
        """
        Wrapper funtion to read labels from file. 
        
        This is not a generic approach but it's required for inference. The main reason is, 
        due to the memory limitation in Git, large files cannot be added. Even though Git LFS 
        can be used but it's not feasible for this current application. So this function is 
        a little play around.

        Returns:
        --------
            labels: List
        """
        with open(f"{self.train_dir}/labels.txt", "r") as file:
            file_data: str = file.read()
            labels: List = file_data.split(",")
            file.close()
            return labels

def convert_audio_to_mfcc(audio_file_path: str, 
                          n_mfcc: int, mfcc_length: int,
                          sampling_rate: int) -> np.ndarray:
    """
    Helper function to convert each audio file to MFCC features. It's 
    a generic function which can be called without an instance.

    Parameters
    ----------
    audio_file_path: str
        Path of audio file.
    n_mfcc: int
        Number of MFCCs to return.
    mfcc_length: int
        Length of MFCC features for each audio input.
    sampling_rate: int
        Target sampling rate.

    Returns
    -------
    mfcc_features: np.ndarray
        Extracted MFCC features of the audio file.
    """
    audio, sampling_rate = librosa.load(audio_file_path, sr = sampling_rate)
    mfcc_features: np.ndarray = librosa.feature.mfcc(audio, n_mfcc = n_mfcc, 
                                                     sr = sampling_rate)
    if(mfcc_length > mfcc_features.shape[1]):
        padding_width = mfcc_length - mfcc_features.shape[1]
        mfcc_features = np.pad(mfcc_features, 
                              pad_width =((0, 0), (0, padding_width)), mode ='constant')
    else:
        mfcc_features = mfcc_features[:, :mfcc_length]
    
    return mfcc_features

def check_fileType(filename: str, extension: str) -> bool:
    """
    Helper function to check the extension of a file.

    Parameters
    ----------
    filename: str
        Input filename
    extension: str
        File extension to check.

    Returns
    -------
        bool: True if a file exists, else False.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extension


def print_shape(name: str, arr: np.array) -> None:
    """
    Helper function to print shapes of input np arrays.

    Note:
        To avoid boilerplate code!!!!

    Parameters
    ----------
    name: str
        Name of input array.
    arr: np.array
        Input array itself.

    Returns
    -------
        None
    """
    print(f"Shape of {name}: {arr.shape}")