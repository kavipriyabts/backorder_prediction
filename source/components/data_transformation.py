import sys
import os

# Dynamically add the root project directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from source.ml.pre_processing import Winsorizer, drop_columns  # ✅ Now should work
from source.constants.training_pipeline import TARGET_COLUMN, SCHEMA_DROP_COLS, SCHEMA_FILE_PATH
from source.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from source.entity.config_entity import DataTransformationConfig
from source.exception import BackOrderException
from source.logger import logging
from source.utils import save_numpy_array_data, save_object, read_yaml_file

class DataTransformation:
    """
    Class for transforming and preparing data for model training.
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        """
        Initialize the DataTransformation instance.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise BackOrderException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read data from the specified file path into a DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise BackOrderException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Get the data transformer object based on schema configuration.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            numerical_cols = list(set(self._schema_config["numerical"]) - set(self._schema_config[SCHEMA_DROP_COLS]))
            categorical_cols = list(set(self._schema_config["categorical"]) - set(self._schema_config[SCHEMA_DROP_COLS]) - {TARGET_COLUMN})

            # Numerical pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('scaler', StandardScaler()),
                ('outlier_clipping', Winsorizer()),
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="most_frequent")),
                ('encoder', OneHotEncoder(drop='first')),
            ])

            # Combining pipelines
            input_preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_cols),
                    ('cat', cat_pipeline, categorical_cols),
                ]
            )

            logging.info("Created preprocessor object by combining numerical and categorical pipelines")
            return input_preprocessor

        except Exception as e:
            raise BackOrderException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiate the data transformation process, including feature engineering and saving artifacts.
        """
        try:
            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()

            # Reading train and test data
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            # Dropping unnecessary features
            train_df = drop_columns(train_df)
            test_df = drop_columns(test_df)

            # Splitting features and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Transforming features
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Transforming target feature
            label_encoder = LabelEncoder()
            target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            # Handling data imbalance
            smt = SMOTETomek(sampling_strategy="minority")
            input_feature_train_arr, target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)

            # Saving transformed data
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            save_object(self.data_transformation_config.preprocessor_object_file_path, preprocessor)
            save_object(self.data_transformation_config.label_encoder_object_file_path, label_encoder)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Saved the preprocessor and label encoder object")

            return DataTransformationArtifact(
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path,
                label_encoder_object_file_path=self.data_transformation_config.label_encoder_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

        except Exception as e:
            raise BackOrderException(e, sys) from e



        
print("Data_transformation file completed")
