import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
import joblib  # Import joblib for loading models
import os

import logging  # Update import if needed
from exception import BackOrderException  # Update import if needed
from ml.pre_processing import drop_columns  # Ensure this function exists

class PredictionPipeline:
    """
    PredictionPipeline handles the prediction process for backorder prediction.

    Methods:
        get_data():
            Retrieve prediction data from a local CSV file.

        get_model():
            Load the trained model from a pickle file.

        predict(model, dataframe):
            Make predictions using the trained model.

        get_labels(model, prediction_array):
            Convert prediction output to human-readable labels.

        initiate_prediction():
            Run the full prediction pipeline.
    """

    def __init__(self, dataset_path="dataset/Kaggle_Test_Dataset_v2.csv", model_path="ml_pipeline.pkl") -> None:
        """
        Initialize the PredictionPipeline.
        """
        try:
            self.dataset_path = dataset_path
            self.model_path = model_path
        except Exception as e:
            raise BackOrderException(f"Error initializing PredictionPipeline: {str(e)}", sys)

    def get_data(self) -> DataFrame:
        """
        Retrieve prediction data from a CSV file.
        """
        try:
            logging.info("Loading prediction data from CSV file")
            prediction_df = pd.read_csv(self.dataset_path)
            
            # Uncomment if you need to drop specific columns
            # prediction_df = drop_columns(prediction_df)
            # logging.info("Dropped specified columns")

            logging.info(f"Prediction DataFrame loaded successfully with shape: {prediction_df.shape}")
            return prediction_df

        except Exception as e:
            raise BackOrderException(f"Error loading prediction data: {str(e)}", sys)

    def get_model(self):
        """
        Load the trained model from a pickle file.
        """
        try:
            logging.info("Loading trained model from pickle file")
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully")
            return model
            
        except Exception as e:
            raise BackOrderException(f"Error loading model: {str(e)}", sys)

    def predict(self, model, dataframe: DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        try:
            logging.info("Making predictions using the trained model")
            predictions = model.predict(dataframe)
            logging.info("Predictions made successfully")
            return predictions

        except Exception as e:
            raise BackOrderException(f"Error in prediction: {str(e)}", sys)

    def get_labels(self, model, prediction_array: np.ndarray) -> np.ndarray:
        """
        Convert prediction output to original labels.
        """
        try:
            logging.info("Converting predictions to original labels")
            labels = model.get_original_labels(prediction_array)
            logging.info("Labels converted successfully")
            return labels

        except Exception as e:
            raise BackOrderException(f"Error in get_labels: {str(e)}", sys)

    def initiate_prediction(self) -> pd.DataFrame:
        """
        Run the prediction pipeline.
        """
        try:
            logging.info("Starting prediction pipeline")

            # Retrieve data
            dataframe = self.get_data()
            
            # Load model
            model = self.get_model()

            # Make predictions
            predicted_arr = self.predict(model, dataframe)

            # Get original labels
            predicted_labels = self.get_labels(model, predicted_arr)

            # Prepare the final DataFrame
            prediction = pd.DataFrame(predicted_labels, columns=["class"])
            predicted_dataframe = pd.concat([dataframe.reset_index(drop=True), prediction], axis=1)

            logging.info("Prediction pipeline completed successfully")

            return predicted_dataframe

        except Exception as e:
            raise BackOrderException(f"Error in initiate_prediction: {str(e)}", sys)
