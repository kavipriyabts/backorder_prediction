import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure 'source' is in sys.path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from source.constants.training_pipeline import SCHEMA_DROP_COLS, SCHEMA_FILE_PATH
from source.entity.artifact_entity import DataIngestionArtifact
from source.entity.config_entity import DataIngestionConfig
from source.exception import BackOrderException
from source.logger import logging
from source.utils import read_yaml_file



class DataIngestion:
    """
    Class for ingesting data from a CSV file and splitting it into training and testing datasets.
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BackOrderException(e, sys)

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """
        try:
            logging.info("Loading data from CSV file")
            file_path = self.data_ingestion_config.input_data_file_path
            dataframe = pd.read_csv(file_path)
            logging.info(f"Shape of loaded dataframe: {dataframe.shape}")
            return dataframe
        except Exception as e:
            raise BackOrderException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split the given dataframe into training and testing datasets and export them.
        """
        logging.info("Splitting data into train and test sets")
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info("Train and test datasets saved successfully")
        except Exception as e:
            raise BackOrderException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process.
        """
        logging.info("Starting data ingestion process")
        try:
            if not (os.path.isfile(self.data_ingestion_config.training_file_path) and 
                    os.path.isfile(self.data_ingestion_config.testing_file_path)):
                dataframe = self.load_data()
                self.split_data_as_train_test(dataframe)
                logging.info("Data successfully ingested and split")
            else:
                logging.info("Dataset already exists. Skipping ingestion.")

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
        except Exception as e:
            raise BackOrderException(e, sys)
print("Data_ingestion file completed0")
