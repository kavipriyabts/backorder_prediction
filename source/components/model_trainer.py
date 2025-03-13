import sys
from source.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from source.entity.config_entity import ModelTrainerConfig
from source.exception import BackOrderException
from source.logger import logging
from source.ml.metric import calculate_metric
from source.ml.estimator import BackOrderPredictionModel
from source.utils import load_numpy_array_data, load_object, save_object
from source.ml.model import TunedModel

class ModelTrainer:
    """
    This class is responsible for training and saving the best predictive model.

    Args:
        data_transformation_artifact (DataTransformationArtifact): Data transformation artifact.
        model_trainer_config (ModelTrainerConfig): Model trainer configuration.

    Methods:
        initiate_model_trainer() -> ModelTrainerArtifact:
            Initiate the model training process and return the model trainer artifact.
    """

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        """
        Initialize ModelTrainer instance.
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            logging.info("ModelTrainer initialized successfully.")
        except Exception as e:
            raise BackOrderException(f"Error initializing ModelTrainer: {str(e)}", sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate the model training process and return the model trainer artifact.
        """
        logging.info("Starting model training process...")

        try:
            # Load transformed training and testing data
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            logging.info(f"Training data shape: {train_arr.shape}, Testing data shape: {test_arr.shape}")

            # Split features and target
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Initialize and train the model
            logging.info("Initializing the Tuned Model for training...")
            model = TunedModel().initiate_model()
            model.fit(x_train, y_train)
            logging.info("Model training completed.")

            # Calculate metrics for training and testing
            logging.info("Calculating performance metrics...")
            model_train_metrics: ClassificationMetricArtifact = calculate_metric(model, x_train, y_train)
            model_test_metrics: ClassificationMetricArtifact = calculate_metric(model, x_test, y_test)

            logging.info(f"Training Balanced Accuracy: {model_train_metrics.balanced_accuracy_score}")
            logging.info(f"Testing Balanced Accuracy: {model_test_metrics.balanced_accuracy_score}")

            # Check if the model meets the expected accuracy
            if model_test_metrics.balanced_accuracy_score < self.model_trainer_config.expected_accuracy:
                logging.error("No best model found with test score more than base score.")
                raise Exception("No best model found with test score more than base score")
            
            logging.info("Model meets expected accuracy requirements.")

            # Load preprocessing and label encoder objects
            preprocessing_obj = load_object(
                file_path=self.data_transformation_artifact.preprocessor_object_path
            )
            label_encoder_obj = load_object(
                file_path=self.data_transformation_artifact.label_encoder_object_path
            )

            # Create the BackOrderPredictionModel object
            logging.info("Creating BackOrderPredictionModel object...")
            backOrder_prediction_model = BackOrderPredictionModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=model,
                label_encoder_object=label_encoder_obj
            )

            # Save the trained model locally
            save_object(self.model_trainer_config.trained_model_file_path, backOrder_prediction_model)
            logging.info(f"Trained model saved at {self.model_trainer_config.trained_model_file_path}")

            # Create and return the model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=model_train_metrics,
                test_metric_artifact=model_test_metrics,
            )

            logging.info("Model training process completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise BackOrderException(f"Error in initiate_model_trainer: {str(e)}", sys)

print("✅ model_trainer.py is completed successfully")
