import pandas as pd
from pandas  import DataFrame
import sys
import os # Add the parent directory of 'source' to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.constants.training_pipeline import SCHEMA_FILE_PATH
from source.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from source.entity.config_entity import DataValidationConfig
from source.exception import BackOrderException
from source.logger import logging
from source.utils import read_yaml_file, write_yaml_file





class DataValidation:
    """
    This class is responsible for validating the data based on a predefined schema.

    Args:
        data_ingestion_artifact (DataIngestionArtifact): Data ingestion artifact.
        data_validation_config (DataValidationConfig): Data validation configuration.

    Raises:
        BackOrderException: If an exception occurs during initialization.

    Methods:
        read_data(file_path: str) -> pd.DataFrame:
            Read data from a CSV file.

        validate_number_of_columns(dataframe: pd.DataFrame) -> bool:
            Validate the number of columns in the dataframe.

        is_numerical_column_exist(df: pd.DataFrame) -> bool:
            Check if numerical columns are present in the dataframe.

        is_categorical_column_exist(df: pd.DataFrame) -> bool:
            Check if categorical columns are present in the dataframe.

        detect_dataset_drift():
            Detect dataset drift.

        initiate_data_validation() -> DataValidationArtifact:
            Initiate data validation.
    """

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initialize DataValidation instance.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise BackOrderException(e, sys)

    @ staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read data from a CSV file.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise BackOrderException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validate the number of columns in the dataframe.
        """

        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])

            logging.info(f"Is required column present: [{status}]")

            return status

        except Exception as e:
            raise BackOrderException(e, sys)

    def is_numerical_column_exist(self, df: DataFrame) -> bool:
        """
        Check if required numerical columns are present in the dataframe.
        """

        try:
            dataframe_columns = df.columns

            status = True

            missing_numerical_columns = []

            for column in self._schema_config["numerical"]:
                if column not in dataframe_columns:
                    status = False

                    missing_numerical_columns.append(column)

            logging.info(f"Missing numerical column: {missing_numerical_columns}")

            return status

        except Exception as e:
            raise BackOrderException(e, sys) from e

    def is_categorical_column_exist(self,df) -> bool:
        """
        Check if required categorical columns are present in the dataframe.
        """
        
        try:
            dataframe_columns = df.columns

            status = True

            missing_categorical_columns = []

            for column in self._schema_config["categorical"]:
                if column not in dataframe_columns:
                    status = False

                    missing_categorical_columns.append(column)

            logging.info(f"Missing numerical column: {missing_categorical_columns}")

            return status

        except Exception as e:
            raise BackOrderException(e, sys) from e

    def detect_dataset_drift(self):
        pass

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiate data validation.
        """

        try:
            validation_error_msg = ""

            logging.info("Starting data validation")

            train_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.trained_file_path
            )

            test_df = DataValidation.read_data(
                file_path=self.data_ingestion_artifact.test_file_path
            )
            
            # validating number of columns
            status = self.validate_number_of_columns(dataframe=train_df)

            logging.info(
                f"All required columns present in training dataframe: {status}"
            )

            if not status:
                validation_error_msg += f"Columns are missing in training dataframe."

            status = self.validate_number_of_columns(dataframe=test_df)

            logging.info(f"All required columns present in testing dataframe: {status}")

            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            
            # checking presence of numerical columns
            status = self.is_numerical_column_exist(df=train_df)

            if not status:
                validation_error_msg += (
                    f"Numerical columns are missing in training dataframe."
                )

            status = self.is_numerical_column_exist(df=test_df)

            if not status:
                validation_error_msg += (
                    f"Numerical columns are missing in test dataframe."
                )
            
            # checking presence of categorical columns
            status = self.is_categorical_column_exist(df=train_df)

            if not status:
                validation_error_msg += (
                    f"Categorical columns are missing in training dataframe."
                )

            status = self.is_categorical_column_exist(df=test_df)

            if not status:
                validation_error_msg += (
                    f"Categorical columns are missing in test dataframe."
                )
            
            # if invalid data encountered stop the pipeline
            validation_status = len(validation_error_msg) == 0
            
            if not validation_status:
                logging.info(f"Validation_error: {validation_error_msg}")
                raise Exception(validation_error_msg)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=None,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact

            

        except Exception as e:
            raise BackOrderException(e, sys) from e  
print("Data_validation file completed")
        