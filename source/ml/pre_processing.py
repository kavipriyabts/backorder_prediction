from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import os
from source.logger import logging  # Correct


class Winsorizer(TransformerMixin):
    def __init__(self, lower_quantile: float = 0.05, upper_quantile: float = 0.95):
        """
        Initialize the Winsorizer transformer.

        Parameters:
        - lower_quantile (float): Lower quantile for winsorization (default: 0.05).
        - upper_quantile (float): Upper quantile for winsorization (default: 0.95).
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y=None) -> 'Winsorizer':
        """
        Fit the Winsorizer transformer.

        Parameters:
        - X (array-like): Input data.
        - y: Ignored.

        Returns:
        - self: Returns the instance of the transformer.
        """
        # Validate input
        if not isinstance(X, (np.ndarray, pd.Series)):
            raise ValueError("Input data must be a numpy array or pandas Series.")

        # Calculate the lower and upper IQR
        Q1 = np.nanpercentile(X, 25)
        Q3 = np.nanpercentile(X, 75)
        IQR = Q3 - Q1

        # Calculate the lower and upper bounds
        self.lower_bound = max(Q1 - (1.5 * IQR), np.nanmin(X))
        self.upper_bound = min(Q3 + (1.5 * IQR), np.nanmax(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using winsorization.

        Parameters:
        - X (array-like): Input data to be transformed.

        Returns:
        - X_transformed (array-like): Transformed data after winsorization.
        """
        if not hasattr(self, 'lower_bound') or not hasattr(self, 'upper_bound'):
            raise RuntimeError("The Winsorizer has not been fitted yet.")

        return np.clip(X, self.lower_bound, self.upper_bound)

    def get_feature_names_out(self, input_features: np.ndarray) -> np.ndarray:
        """
        Get the feature names after transformation.

        Parameters:
        - input_features (array-like): Input feature names.

        Returns:
        - output_features (array-like): Transformed feature names.
        """
        return input_features

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns before data transformation.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame after dropping specified columns.
    """
    # Define the columns to drop directly instead of reading from YAML
    columns_to_drop = ["column1", "column2", "column3"]  # Replace with actual column names

    df = df.drop(columns_to_drop, axis=1, errors="ignore")  # Avoid KeyErrors

    logging.info(f"Features dropped: {columns_to_drop}")

    return df
