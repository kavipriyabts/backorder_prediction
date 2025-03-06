import os
import sys
import dill
import numpy as np
import yaml

# Corrected imports with source prefix
from source.exception import BackOrderException  # ✅ Corrected
from source.logger import logging  # ✅ Corrected



def read_yaml_file(file_path: str) -> dict:
    """
    Reads content from a YAML file and returns it as a dictionary.
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise BackOrderException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise BackOrderException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads an object from a file using the dill library.
    """
    logging.info("Entered the load_object method")

    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method")
        return obj
    except Exception as e:
        raise BackOrderException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save numpy array data to file.
    
    :param file_path: str location of file to save
    :param array: np.ndarray data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise BackOrderException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file.
    
    :param file_path: str location of file to load
    :return: np.ndarray data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise BackOrderException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object to a file using the dill library.
    """
    logging.info("Entered the save_object method")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method")
    except Exception as e:
        raise BackOrderException(e, sys) from e


print("utile file completed")