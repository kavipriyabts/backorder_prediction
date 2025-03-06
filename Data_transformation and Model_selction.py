import os
import warnings
import gc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, roc_curve, auc, confusion_matrix)
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import optuna
import pickle
from tabulate import tabulate

# Settings
warnings.filterwarnings("ignore")

# Function to downcast numeric types
def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# Function to load data in chunks
def load_data_in_chunks(file_path: str, chunk_size: int) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk = downcast_dtypes(chunk)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Custom functions and classes
class Winsorizer:
    def __init__(self, change=True):
        self.change = change

    def fit(self, X, y=None):
        Q1 = np.nanpercentile(X, 25)
        Q3 = np.nanpercentile(X, 75)
        IQR = Q3 - Q1
        self.lower_bound = Q1 - 1.5 * IQR
        self.upper_bound = Q3 + 1.5 * IQR
        return self

    def transform(self, X):
        if not self.change:
            return X
        return np.clip(X, self.lower_bound, self.upper_bound)

class CustomMLPipeline:
    useless_cols = ['in_transit_qty', 'local_bo_qty', 'pieces_past_due', 'potential_issue', 'oe_constraint', 'rev_stop']
    sparse_cols = ['in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'local_bo_qty']
    numeric_cols = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month', 'forecast_6_month', 'forecast_9_month', 'sales_1_month', 'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank', 'pieces_past_due', 'perf_6_month_avg', 'perf_12_month_avg', 'local_bo_qty']
    cat_cols = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']

    def __init__(self, imputer_type, drop_and_winsorize, classifier, columns_to_drop, add_indicator=False, keep_first=False):
        self.imputer_type = imputer_type
        self.drop_and_winsorize = drop_and_winsorize
        self.classifier = classifier
        self.columns_to_drop = columns_to_drop
        self.add_indicator = add_indicator
        self.keep_first = keep_first

        if self.drop_and_winsorize:
            self.drop_cols = self.useless_cols if self.columns_to_drop == 'useless' else self.sparse_cols
        else:
            self.drop_cols = []

        self.num_cols_used = [col for col in self.numeric_cols if col not in self.drop_cols]
        self.cat_cols_used = [col for col in self.cat_cols if col not in self.drop_cols]

    def _create_classifier(self):
        if self.classifier == 'RandomForest':
            return RandomForestClassifier()
        elif self.classifier == 'SVC':
            return SVC()
        elif self.classifier == 'LogisticRegression':
            return LogisticRegression(solver='liblinear')
        elif self.classifier == 'XGBoost':
                        return XGBClassifier()
        else:
            raise ValueError(f"Invalid classifier: {self.classifier}")

    def _drop_col(self, X):
    # Check if 'sku' is in the DataFrame columns before trying to drop it
        if 'sku' in X.columns:
            X.drop('sku', axis=1, inplace=True)

        # Drop other specified columns
        columns_to_drop = self.drop_cols
        return X.drop(columns_to_drop, axis=1, errors='ignore')  # Use errors='ignore' to avoid KeyError
    def create_pipeline(self):
        # Conditional creation of imputer based on 'imputer_type'
        if self.imputer_type == 'knn':
            imp = KNNImputer(weights='distance', add_indicator=bool(self.add_indicator))  # Ensure it's a boolean
        else:
            imp = SimpleImputer(strategy='median', add_indicator=bool(self.add_indicator))  # Ensure it's a boolean

        # Conditional initialization of cat_encoder based on 'keep_first'
        cat_encoder = OneHotEncoder(drop='first') if self.keep_first else OneHotEncoder()

        # Constructing the numerical pipeline
        num_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', imp),
            ('outlier_clipping', Winsorizer(change=self.drop_and_winsorize)),
        ])

        # Constructing the categorical pipeline
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehotencoder', cat_encoder),
        ])

        # Constructing the training pipeline
        clf = self._create_classifier()
        training_pipeline = Pipeline([
            ('Drop_Columns', FunctionTransformer(self._drop_col)),
            ('Balancing', RandomOverSampler()),
            ('Feature_transform', ColumnTransformer([
                ('num_pipeline', num_pipeline, self.num_cols_used),
                ('cat_pipeline', cat_pipeline, self.cat_cols_used),
            ])),
            ('model', clf)
        ])

        return training_pipeline

# Function to evaluate and plot results
def evaluate_and_plot(y_train, y_val, y_train_pred, y_train_pred_proba, y_val_pred, y_val_pred_proba):
    performance_metric = [balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score]
    report = []

    for metric in performance_metric:
        if metric == roc_auc_score:
            train_score = metric(y_train, y_train_pred_proba)
            val_score = metric(y_val, y_val_pred_proba)
        else:
            train_score = metric(y_train, y_train_pred)
            val_score = metric(y_val, y_val_pred)

        report.append([metric.__name__, f'{train_score:.2f}', f'{val_score:.2f}'])

    report_table = tabulate(report, headers=["Metric", "Train Score", "Validation Score"], tablefmt="pretty")
    print('Model evaluation report:\n')
    print(report_table)

    conf_matrix = confusion_matrix(y_val, y_val_pred, normalize='true') * 100
    headers = ["", "Predicted Negative", "Predicted Positive"]
    rows = [
        ["Actual Negative", f"TNR = {conf_matrix[0][0]:.2f}", f"FPR = {conf_matrix[0][1]:.2f}"],
        ["Actual Positive", f"FNR = {conf_matrix[1][0]:.2f}", f"TPR = {conf_matrix[1][1]:.2f}"]
    ]
    print('\nValidation Confusion matrix:\n')
    print(tabulate(rows, headers=headers, tablefmt="pretty"))
    print('\n')

    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
    auc_val = auc(fpr_val, tpr_val)

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
    auc_train = auc(fpr_train, tpr_train)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='green', label=f'Train AUC = {auc_train:.2f}')
    plt.plot(fpr_val, tpr_val, color='blue', label=f'Validation AUC = {auc_val:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Train & Validation')
    plt.legend()
    plt.show()



# Get the folder where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create the dataset path dynamically
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Kaggle_Training_Dataset_v2.csv")

# Use this path to load data
chunk_size = 100000
data = load_data_in_chunks(DATASET_PATH, chunk_size)


# Data cleaning
data.dropna(subset=data.columns.drop('lead_time'), inplace=True)

# Split features and target
y = LabelEncoder().fit_transform(data['went_on_backorder'])
X = data.drop('went_on_backorder', axis=1)

# Print shapes
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Split the data into train and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_val:", y_val.shape)

# Define the best pipeline parameters
best_pipeline_param = {
    'classifier': 'XGBoost',  # You can change this to any classifier you want to test
    'drop_and_winsorize': 1,
    'columns_to_drop': 'useless',
    'add_indicator': True,  # Ensure this is a boolean
    'imputer_type': 'simple',
    'keep_first': True
}

# Create the ML pipeline
best_ml_pipeline = CustomMLPipeline(**best_pipeline_param).create_pipeline()

# Fit the pipeline to the training data
print('Training begins...')
best_ml_pipeline.fit(X_train, y_train)
print('Training finished.')

# Predict training labels and probabilities
y_train_pred = best_ml_pipeline.predict(X_train)
y_train_pred_proba = best_ml_pipeline.predict_proba(X_train)[:, 1]

# Predict validation labels and probabilities
y_val_pred = best_ml_pipeline.predict(X_val)
y_val_pred_proba = best_ml_pipeline.predict_proba(X_val)[:, 1]

# Evaluate and plot results
evaluate_and_plot(y_train, y_val, y_train_pred, y_train_pred_proba, y_val_pred, y_val_pred_proba)

# Save the trained pipeline as a pickle file
with open('ml_pipeline.pkl', 'wb') as file:
    pickle.dump(best_ml_pipeline, file)

# Load the trained pipeline from the pickle file
with open('ml_pipeline.pkl', 'rb') as file:
    loaded_ml_pipeline = pickle.load(file)

# Example of using the loaded model for predictions
# Assuming you have a new dataset for testing
# X_test = pd.read_csv('path_to_your_test_file.csv')  # Load your test data
# y_test = LabelEncoder().fit_transform(X_test['went_on_backorder'])
# X_test = X_test.drop('went_on_backorder', axis=1)

# Predict test labels and probabilities
# y_test_pred = loaded_ml_pipeline.predict(X_test)
# y_test_pred_proba = loaded_ml_pipeline.predict_proba(X_test)[:, 1]

# Evaluate the loaded model (if you have test data)
# evaluate_and_plot(y_test, y_test, y_test_pred, y_test_pred_proba, y_test_pred, y_test_pred_proba)

# Clean up memory
gc.collect()