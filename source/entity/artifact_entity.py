from dataclasses import dataclass

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    recall_score: float
    precision_score: float
    balanced_accuracy_score: float
    roc_auc_score: float

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_train_file_path: str
    validated_test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    model_accuracy: float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    best_model_path: str
    best_model_accuracy: float
    current_model_path: str
    current_model_accuracy: float

@dataclass
class ModelPusherArtifact:
    saved_model_path: str
