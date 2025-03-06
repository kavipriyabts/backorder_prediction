from source.exception import BackOrderException
from source.logger import logging
from source.entity.artifact_entity import (
    DataValidationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    DataTransformationArtifact,
    ClassificationMetricArtifact
)
from source.entity.config_entity import ModelEvaluationConfig
import os
import sys
import pandas as pd
from typing import Optional
from source.ml.metric import calculate_metric
from source.ml.estimator import BackOrderPredictionModel
from source.utils import save_object, load_object
from source.constants.training_pipeline import TARGET_COLUMN

class ModelEvaluation:
    """
    This class is responsible for evaluating and comparing the trained model's performance.

    Args:
        model_eval_config (ModelEvaluationConfig): Model evaluation configuration.
        data_validation_artifact (DataValidationArtifact): Data validation artifact.
        data_transformation_artifact (DataTransformationArtifact): Data transformation artifact.
        model_trainer_artifact (ModelTrainerArtifact): Model trainer artifact.

    Methods:
        evaluate_model() -> bool:
            Evaluate the performance of the trained model and compare it with the best model.

        initiate_model_evaluation() -> ModelEvaluationArtifact:
            Initiate the model evaluation process and return the model evaluation artifact.
    """

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initialize ModelEvaluation instance.
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise BackOrderException(e, sys)

    def evaluate_model(self) -> bool:
        """
        Evaluate the performance of the trained model and compare it with the best model.
        """
        try:
            # Load the test data
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            # Split features and target
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # Load the trained model and label encoder
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            label_encoder = load_object(file_path=self.data_transformation_artifact.label_encoder_object_file_path)

            # Transform the target variable
            y = label_encoder.transform(y)

            # Calculate metrics for the trained model
            trained_model_score: ClassificationMetricArtifact = calculate_metric(trained_model, x, y)
            trained_model_balanced_accuracy = trained_model_score.balanced_accuracy_score

            # For this example, we will not compare with a best model
            # Instead, we will just log the trained model's performance
            logging.info(f"Trained model balanced accuracy = {trained_model_balanced_accuracy}")

            # Define a threshold for model acceptance
            is_model_accepted = trained_model_balanced_accuracy > self.model_eval_config.expected_accuracy_threshold

            return is_model_accepted

        except Exception as e:
            raise BackOrderException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiate the model evaluation process and return the model evaluation artifact.
        """
        try:
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=self.evaluate_model()
            )

            return model_evaluation_artifact

        except Exception as e:
            raise BackOrderException(e, sys)
print("model_evaluation file is completed")