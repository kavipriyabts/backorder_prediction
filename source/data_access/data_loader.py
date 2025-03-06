import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from CSV."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
        return pd.read_csv(self.data_path)
