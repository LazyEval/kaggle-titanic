from pathlib import Path
import pandas as pd


class DataSet:
    def __init__(self, train_dir, test_dir):
        """Set train and test set directory attributes."""
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)

    def get_train_set(self):
        """Create train pandas DataFrame from raw data file."""
        return pd.read_csv(self.train_dir)

    def get_test_set(self):
        """Create test pandas DataFrame from raw data file."""
        return pd.read_csv(self.test_dir)

    @staticmethod
    def get_features(data, target=None):
        """Return DataFrame of features for modeling."""
        if target:
            features = data.columns.drop([target, 'PassengerId']).values
        else:
            features = data.columns.drop(['PassengerId']).values
        return data[features]

    @staticmethod
    def get_label(data, target='Survived'):
        """Return label Series for modeling."""
        return data[target]
