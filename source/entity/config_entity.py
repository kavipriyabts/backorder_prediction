from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    input_data_file_path: str = "path/to/dataset.csv"
    train_test_split_ratio: float = 0.2
    training_file_path: str = "path/to/train.csv"
    testing_file_path: str = "path/to/test.csv"


@dataclass
class DataValidationConfig:
    schema_file_path: str
    validated_train_file_path: str
    validated_test_file_path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.validated_train_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.validated_test_file_path), exist_ok=True)

@dataclass
class DataTransformationConfig:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.transformed_train_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.transformed_test_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.preprocessor_object_path), exist_ok=True)

@dataclass
class ModelTrainerConfig:
    trained_model_path: str
    expected_accuracy: float

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.trained_model_path), exist_ok=True)

@dataclass
class ModelEvaluationConfig:
    best_model_path: str
    best_model_accuracy: float
    current_model_path: str
    current_model_accuracy: float

@dataclass
class ModelPusherConfig:
    saved_model_path: str

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.saved_model_path), exist_ok=True)

