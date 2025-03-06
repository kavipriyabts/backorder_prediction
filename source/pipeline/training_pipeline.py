import sys
import os
import source
import pandas as pd

from source.data_access.data_loader import DataLoader
from source.components.data_transformation import DataTransformation
from source.components.data_validation import DataValidation
from source.components.model_evaluation import ModelEvaluation
from source.components.model_pusher import ModelPusher
from source.components.model_trainer import ModelTrainer
from source.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from source.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
    ModelTrainerConfig,
)
from source.exception import BackOrderException
from source.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion from CSV file")
            data_loader = DataLoader("dataset/Kaggle_Training_Dataset_v2.csv")
            df = data_loader.load_data()
            logging.info("Data ingestion completed successfully")
            return df  # Directly returning the DataFrame instead of an artifact
        except Exception as e:
            raise BackOrderException(e, sys) from e

    def start_data_validation(self, df: pd.DataFrame) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation = DataValidation(df, self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise BackOrderException(e, sys) from e

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_validation_artifact, self.data_transformation_config
            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed successfully")
            return data_transformation_artifact
        except Exception as e:
            raise BackOrderException(e, sys)

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            logging.info("Starting model training")
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Model training completed successfully")
            return model_trainer_artifact
        except Exception as e:
            raise BackOrderException(e, sys)

    def start_model_evaluation(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation")
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                data_validation_artifact=data_validation_artifact,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            logging.info("Model evaluation completed successfully")
            return model_evaluation_artifact
        except Exception as e:
            raise BackOrderException(e, sys)

    def start_model_pusher(self, model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info("Starting model pusher")
            model_pusher = ModelPusher(
                model_pusher_config=self.model_pusher_config,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("Model pusher completed successfully")
            return model_pusher_artifact
        except Exception as e:
            raise BackOrderException(e, sys)

    def run_pipeline(self):
        try:
            logging.info("Starting the training pipeline")
            df = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(df)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(
                data_validation_artifact, data_transformation_artifact, model_trainer_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted. Exiting the training pipeline.")
                return None

            model_pusher_artifact = self.start_model_pusher(model_trainer_artifact)
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            raise BackOrderException(e, sys) from e
