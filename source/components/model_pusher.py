import sys
import os
from source.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from source.entity.config_entity import ModelPusherConfig
from source.exception import BackOrderException
from source.logger import logging

class ModelPusher:
    """
    This class is responsible for saving the trained model locally.

    Args:
        model_trainer_artifact (ModelTrainerArtifact): Model trainer artifact.
        model_pusher_config (ModelPusherConfig): Model pusher configuration.

    Methods:
        initiate_model_pusher() -> ModelPusherArtifact:
            Save the trained model to the local filesystem and return the model pusher artifact.
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        model_pusher_config: ModelPusherConfig,
    ):
        """
        Initialize ModelPusher instance.
        """
        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Save the trained model to the local filesystem.
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")

        try:
            # Define the local path where the model will be saved
            local_model_path = os.path.join(self.model_pusher_config.local_model_dir, "model.pkl")

            logging.info("Saving model to local path: %s", local_model_path)

            # Save the trained model to the local filesystem
            # Assuming you have a function to save the model, e.g., save_object
            from source.utils import save_object
            save_object(local_model_path, self.model_trainer_artifact.trained_model_file_path)

            model_pusher_artifact = ModelPusherArtifact(
                local_model_path=local_model_path,
            )

            logging.info("Model saved successfully to local path: %s", local_model_path)
            logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")

            return model_pusher_artifact

        except Exception as e:
            raise BackOrderException(e, sys) from e
        
print("model_pusher file is completed")