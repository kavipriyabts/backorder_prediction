import sys
import os
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from source.constants.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR
from source.exception import BackOrderException
from source.logger import logging
import numpy as np


class BackOrderPredictionModel:
    """
    This class encapsulates the essential components required for predicting back-order status.

    Args:
        preprocessing_object (Pipeline): A data preprocessing pipeline.
        trained_model_object (object): A trained prediction model.
        label_encoder_object (object): An object for label encoding.

    Methods:
        predict(dataframe: DataFrame) -> np.ndarray:
            Utilizes the trained model to predict back-order status for a given DataFrame.

        get_original_labels(prediction_array: np.ndarray) -> np.ndarray:
            Converts predicted label indices back to their original labels using the label encoder.

        __repr__() -> str:
            Provides a string representation of the class.

        __str__() -> str:
            Offers a human-readable string representation of the class.
    """

    def __init__(self, preprocessing_object: Pipeline, 
                 trained_model_object: object,
                 label_encoder_object: object) -> None:
        """
        Initialize the BackOrderPredictionModel instance.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
        self.label_encoder_object = label_encoder_object

    def predict(self, dataframe: DataFrame) -> np.ndarray:
        """
        Utilizes the trained model to predict back-order status for a given DataFrame.

        Args:
            dataframe (DataFrame): The input DataFrame containing features for prediction.

        Returns:
            np.ndarray: The predicted back-order statuses.
        """
        logging.info("Entered predict method of BackOrderPredictionModel class")

        try:
            logging.info("Using the trained model to get predictions")
            transformed_feature = self.preprocessing_object.transform(dataframe)
            predictions = self.trained_model_object.predict(transformed_feature)

            logging.info(f"Predictions made successfully: {predictions}")
            return predictions

        except Exception as e:
            raise BackOrderException(f"Error in predict: {str(e)}", sys) from e
        
    def get_original_labels(self, prediction_array: np.ndarray) -> np.ndarray:
        """
        Converts predicted label indices back to their original labels using the label encoder.

        Args:
            prediction_array (np.ndarray): The array of predicted label indices.

        Returns:
            np.ndarray: The original labels corresponding to the predicted indices.
        """
        try:
            logging.info('Entered get_original_labels method of BackOrderPredictionModel class')
            original_labels = self.label_encoder_object.inverse_transform(prediction_array)
            logging.info(f"Original labels retrieved: {original_labels}")
            return original_labels
        
        except Exception as e:
                        raise BackOrderException(f"Error in get_original_labels: {str(e)}", sys) from e
        

    def __repr__(self) -> str:
        """
        Provides a string representation of the class.
        """
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        """
        Offers a human-readable string representation of the class.
        """
        return f"{type(self.trained_model_object).__name__}()"