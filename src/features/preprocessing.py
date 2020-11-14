from pathlib import Path
import pandas as pd


class PreProcessing:
    def __init__(self, data):
        """Set data attribute to None."""
        self.data = data

    def impute(self, col, strategy='median'):
        """Impute data from a feature of a DataFrame using either the median or the mode."""
        if strategy == 'median':
            self.data[col] = self.data[col].fillna(self.data[col].dropna().median())
        elif strategy == 'mode':
            self.data[col] = self.data[col].fillna(self.data[col].dropna().mode()[0])

    def make_bands(self, continuous_col, band_col, n_bands, method='standard'):
        """Create bands from a continuous variable of a pandas Series.

        Parameters
        ----------
        continuous_col : str
            Column of continuous variable.
        band_col : str
            Name of new column with bands.
        n_bands : int
            Number of bands desired.
        method : str
            Choice of method for the cut: standard or quantile.

        Returns
        -------
        pandas DataFrame
        """
        if method == 'standard':
            self.data[band_col] = pd.cut(self.data[continuous_col], n_bands)
        elif method == 'quantile':
            self.data[band_col] = pd.qcut(self.data[continuous_col], n_bands)
        else:
            print('Please choose a valid method.')

    def label_encoder(self, col):
        """Label encode a feature of a pandas DataFrame and return the DataFrame."""
        self.data[col] = self.data[col].astype('category').cat.codes

    def one_hot_encoder(self, cols):
        """One-hot encode features of a pandas DataFrame and return the DataFrame."""
        self.data = pd.get_dummies(self.data, columns=cols)


class TitanicPreProcessing(PreProcessing):

    label_cols = ['AgeBand', 'FareBand']
    one_hot_cols = ['Title', 'Sex', 'Embarked']
    drop_cols = ['Name', 'Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']

    def __init__(self, data, file_path):
        """Set data attribute and run pre-processing steps."""
        super().__init__(data)
        self._apply_preprocessor()
        self._output_data(file_path)

    def age_impute(self):
        """Impute age based on median age for every combination of sex and class."""
        for i, sex in enumerate(self.data['Sex'].unique()):
            for j in range(self.data['Pclass'].nunique()):
                guess = self.data.query('Sex==@sex & Pclass==@j+1')['Age'].dropna().median()
                self.data.loc[
                    (self.data['Age'].isnull()) & (self.data['Sex'] == sex) & (self.data['Pclass'] == j + 1), 'Age'] = \
                    int(guess / 0.5 + 0.5) * 0.5  # round up
        self.data['Age'] = self.data['Age'].astype(int)  # convert data type to int

    def create_title(self):
        """Create a Title feature based on the Name feature and replace the titles with the most common instances."""
        self.data['Title'] = (self.data['Name']
                              .str.extract(r' ([A-Za-z]+)\.', expand=False)
                              .replace(['Capt', 'Col', 'Countess', 'Don', 'Dr', 'Jonkheer',
                                        'Lady', 'Major', 'Rev', 'Sir', 'Dona'], 'Rare')
                              .replace('Mlle', 'Miss')
                              .replace('Mme', 'Mrs')
                              .replace('Ms', 'Miss'))

    def create_family(self):
        """Create a family feature based on the 'SibSp' and 'Parch' features."""
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1

    def create_is_alone(self, func=create_family):
        """Create an "is alone" feature based on the family feature."""
        func(self)
        self.data['IsAlone'] = 0
        self.data.loc[self.data['FamilySize'] == 1, 'IsAlone'] = 1

    def _apply_preprocessor(self):
        """Apply all pre-processing steps defined in the PreProcessing and TitanicPreProcessing classes."""
        self.data = self.data.drop(['Ticket', 'Cabin'], axis=1)
        self.age_impute()
        self.impute('Embarked', strategy='mode')
        self.impute('Fare', strategy='median')
        self.create_title()
        self.create_is_alone()
        self.make_bands('Age', 'AgeBand', 5)
        self.make_bands('Fare', 'FareBand', 4, 'quantile')

        for col in self.label_cols:
            self.label_encoder(col)

        self.one_hot_encoder(self.one_hot_cols)
        self.data = self.data.drop(self.drop_cols, axis=1)

    def _output_data(self, file_path):
        """Create data set .csv output file from cleaned DataFrame."""
        self.data.to_csv(Path(file_path), index=False)
